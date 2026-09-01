"""Diccionario K-65P ↔ humano (DL-014) — base de datos de traducción bidireccional.

La base de datos del léxico de Bit: cada palabra humana con su explicación NSM
(65 trits + árbol), su superficie en EN/ES, su estado de descomposición, y las
relaciones semánticas. Es la fuente de verdad del traductor EN↔K-65P y el
aterrizaje del programa de descomposición NSM (nsm_explication.py).

Estados de una palabra:
  canonical    — glifo diseñado por el operador (los 28)
  molecule     — molécula k65p (los 99 del mundo semántico)
  prime        — uno de los 65 primos
  explicated   — descompuesto por el programa NSM (curado)
  drafted      — generado por LLM, pendiente de curación
  pending      — en el vocabulario, sin descomponer aún

Uso:
  d = K65PDictionary("storage/k65p_dictionary.db")
  d.upsert_word("dog", glyph=..., explication={...}, status="explicated")
  d.en_to_k65p("dog")          # → fila con glifo/explicación
  d.k65p_to_en(glyph)          # → candidatos por solape de primos
  d.stats()                    # progreso de la descomposición
"""
import hashlib
import json
import os
import sqlite3

from src.bitnet.vocab.nsm_explication import explication_to_glyph, glyph_to_features

SCHEMA = """
CREATE TABLE IF NOT EXISTS words (
	word_id   INTEGER PRIMARY KEY,
	surface   TEXT NOT NULL UNIQUE,       -- forma canónica EN
	glyph     BLOB,                       -- 65 trits (int8) — el ADN semántico
	explication TEXT,                    -- árbol NSM en JSON (la definición estructurada)
	status    TEXT NOT NULL DEFAULT 'pending',  -- canonical/molecule/prime/explicated/drafted/pending
	source    TEXT,                       -- de dónde salió la descomposición
	curated_by TEXT,
	notes     TEXT,
	updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS surfaces (
	surface  TEXT NOT NULL,
	word_id  INTEGER NOT NULL REFERENCES words(word_id),
	lang     TEXT NOT NULL DEFAULT 'en',
	PRIMARY KEY (surface, lang)
);
CREATE TABLE IF NOT EXISTS primes (
	idx      INTEGER PRIMARY KEY,
	name_es  TEXT NOT NULL,
	name_en  TEXT
);
CREATE TABLE IF NOT EXISTS relations (
	word_a INTEGER NOT NULL REFERENCES words(word_id),
	word_b INTEGER NOT NULL REFERENCES words(word_id),
	relation TEXT NOT NULL,              -- synonym/hyponym/derivation/antonym
	weight REAL DEFAULT 1.0,
	PRIMARY KEY (word_a, word_b, relation)
);
CREATE TABLE IF NOT EXISTS phrases (
	phrase_id INTEGER PRIMARY KEY AUTOINCREMENT,
	span      TEXT NOT NULL,            -- la expresión humana multi-palabra
	lang      TEXT NOT NULL DEFAULT 'en',
	word_id   INTEGER REFERENCES words(word_id),  -- la molécula/concepto K-65P que la comprime
	glyph     BLOB,                     -- o su propio glifo si es concepto nuevo
	context_note TEXT,                  -- cuándo aplica esta compresión (p.ej. phrasal verb)
	source    TEXT,
	UNIQUE (span, lang)
);
CREATE INDEX IF NOT EXISTS idx_words_status ON words(status);
"""


class K65PDictionary:
	def __init__(self, db_path: str):
		os.makedirs(os.path.dirname(db_path), exist_ok=True)
		self.conn = sqlite3.connect(db_path)
		self.conn.executescript(SCHEMA)
		self.conn.commit()

	# ── escritura ──
	def upsert_prime(self, idx: int, name_es: str, name_en: str = None) -> None:
		self.conn.execute(
			"INSERT INTO primes (idx, name_es, name_en) VALUES (?,?,?) "
			"ON CONFLICT(idx) DO UPDATE SET name_es=excluded.name_es, name_en=excluded.name_en",
			(idx, name_es, name_en))
		self.conn.commit()

	def upsert_word(self, surface: str, glyph=None, explication: dict | list | None = None,
					status: str = "pending", source: str = None, surfaces_es: list = None,
					curated_by: str = None) -> int:
		glyph_blob = None
		if glyph is not None:
			import numpy as np
			glyph_blob = np.asarray(glyph, dtype=np.int8).tobytes()
		exp_json = json.dumps(explication, ensure_ascii=False) if explication is not None else None
		cur = self.conn.execute(
			"INSERT INTO words (surface, glyph, explication, status, source, curated_by) "
			"VALUES (?,?,?,?,?,?) "
			"ON CONFLICT(surface) DO UPDATE SET glyph=COALESCE(excluded.glyph, glyph), "
			"explication=COALESCE(excluded.explication, explication), "
			"status=excluded.status, source=excluded.source, curated_by=excluded.curated_by, "
			"updated_at=datetime('now')",
			(surface, glyph_blob, exp_json, status, source, curated_by))
		word_id = cur.lastrowid
		if surfaces_es:
			for s in surfaces_es:
				self.conn.execute(
					"INSERT OR IGNORE INTO surfaces (surface, word_id, lang) VALUES (?,?, 'es')",
					(s, word_id))
		self.conn.commit()
		return word_id

	def add_relation(self, a: str, b: str, relation: str, weight: float = 1.0) -> None:
		ia, ib = self._id(a), self._id(b)
		if ia and ib:
			self.conn.execute(
				"INSERT OR IGNORE INTO relations (word_a, word_b, relation, weight) VALUES (?,?,?,?)",
				(ia, ib, relation, weight))
			self.conn.commit()

	def _id(self, surface: str):
		r = self.conn.execute("SELECT word_id FROM words WHERE surface=?", (surface,)).fetchone()
		return r[0] if r else None

	# ── lectura (el traductor) ──
	def en_to_k65p(self, surface: str) -> dict | None:
		r = self.conn.execute(
			"SELECT word_id, surface, glyph, explication, status, source FROM words WHERE surface=?",
			(surface,)).fetchone()
		if not r:
			return None
		out = {"word_id": r[0], "surface": r[1], "status": r[4], "source": r[5]}
		if r[2]:
			import numpy as np
			out["glyph"] = np.frombuffer(r[2], dtype=np.int8)
		out["explication"] = json.loads(r[3]) if r[3] else None
		return out

	def k65p_to_en(self, glyph) -> list[dict]:
		"""Candidatos humanos para una composición: por solape de primos activos
		(Jaccard), ordenado — el decompilador v0 del RFC-002 §5."""
		import numpy as np
		g = np.asarray(glyph, dtype=np.int8)
		ga = set(np.nonzero(g)[0].tolist())
		best = []
		for wid, surface, blob in self.conn.execute(
				"SELECT word_id, surface, glyph FROM words WHERE glyph IS NOT NULL"):
			gb = set(np.nonzero(np.frombuffer(blob, dtype=np.int8))[0].tolist())
			jac = len(ga & gb) / max(1, len(ga | gb))
			best.append((jac, surface, wid))
		best.sort(reverse=True)
		return [{"surface": s, "overlap": round(j, 3), "word_id": w} for j, s, w in best[:10]]

	def upsert_phrase(self, span: str, word_surface: str = None, glyph=None,
					  context_note: str = None, source: str = None) -> int:
		"""Compresión de span humano → concepto K-65P (la unidad es el CONCEPTO,
		no la palabra: N palabras → 1 molécula/glyph)."""
		word_id = self._id(word_surface) if word_surface else None
		glyph_blob = None
		if glyph is not None:
			import numpy as np
			glyph_blob = np.asarray(glyph, dtype=np.int8).tobytes()
		cur = self.conn.execute(
			"INSERT INTO phrases (span, lang, word_id, glyph, context_note, source) "
			"VALUES (?,?,?,?,?,?) ON CONFLICT(span, lang) DO UPDATE SET "
			"word_id=excluded.word_id, glyph=COALESCE(excluded.glyph, glyph), "
			"context_note=excluded.context_note",
			(span.strip().lower(), "en", word_id, glyph_blob, context_note, source))
		self.conn.commit()
		return cur.lastrowid

	def longest_span_match(self, tokens: list[str]) -> dict | None:
		"""Traducción consciente del contexto: busca el SPAN más largo que
		cubra el inicio de los tokens — la compresión N→1 del diccionario."""
		for n in range(min(len(tokens), 6), 0, -1):
			span = " ".join(tokens[:n]).lower()
			r = self.conn.execute(
				"SELECT p.span, p.word_id, p.glyph, p.context_note, w.surface "
				"FROM phrases p LEFT JOIN words w ON p.word_id = w.word_id "
				"WHERE p.span = ?", (span,)).fetchone()
			if r:
				return {"span": r[0], "word": r[4], "glyph": r[2], "note": r[3],
						"consumed": n}
		return None

	def pending(self, limit: int = None) -> list[str]:
		q = "SELECT surface FROM words WHERE status='pending'"
		q += f" LIMIT {limit}" if limit else ""
		return [r[0] for r in self.conn.execute(q)]

	def stats(self) -> dict:
		out = {}
		for status, n in self.conn.execute("SELECT status, COUNT(*) FROM words GROUP BY status"):
			out[status] = n
		out["total"] = sum(out.values())
		out["relations"] = self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
		return out
