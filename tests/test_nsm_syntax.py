"""Tests de K-65P v0 (Kernel de 65 Primos), la sintaxis canónica de glifos (RFC-002)."""

import pytest

from src.bitnet.translation.nsm_syntax import (
	NSMSyntaxError,
	is_valid,
	linearize,
	parse,
	validate,
)

# Los 20 ejemplos compilados a mano del RFC-002 §3 (corpus semilla).
EJEMPLOS_RFC = [
	"[ver yo algo]",
	"[oír tú agua]",
	"[caliente fuego]",
	"[malo [G fuego caliente]]",
	"[si [tocar alguien [G fuego caliente]] [pasar [G algo malo] alguien]]",
	"[porque [mover tú lejos] [sentir yo malo]]",
	"[querer yo agua]",
	"[querer yo [hacer yo beber agua]]",
	"[quizá [pasar lluvia]]",
	"[poder [mover yo]]",
	"[no [ver yo algo]]",
	"[saber yo [bueno agua]]",
	"[querer yo [decir tú verdad yo]]",
	"[vivir gente aquí]",
	"[antes [pequeño yo]]",
	"[después [pasar [G algo malo]]]",
	"[como [G cosa este] agua]",
	"[morir [G gente todo]]",
	"[porque [pensar yo] [existir yo]]",
	"[si [tocar tú [G agua frío]] [sentir tú frío]]",
]


@pytest.mark.parametrize("ejemplo", EJEMPLOS_RFC)
def test_ejemplos_rfc_son_validos(ejemplo):
	assert validate(ejemplo) == []


@pytest.mark.parametrize("ejemplo", EJEMPLOS_RFC)
def test_ida_y_vuelta_canonica(ejemplo):
	# La forma canónica debe ser un punto fijo: parse → linearize → parse.
	assert linearize(parse(ejemplo)) == ejemplo


def test_aridad_insuficiente():
	assert not is_valid("[sentir yo]")  # sentir exige experimentador + estado


def test_aridad_excesiva():
	assert not is_valid("[morir yo tú]")  # morir es estrictamente unario


def test_operador_desconocido():
	errores = validate("[volar yo]")
	assert any("desconocido" in e for e in errores)


def test_conector_binario_con_un_argumento():
	assert not is_valid("[si [ver yo algo]]")


def test_unario_sobre_atomo():
	# quizá/no/poder operan sobre cláusulas, no sobre átomos sueltos.
	assert not is_valid("[quizá lluvia]")


def test_grupo_sin_nucleo():
	errores = validate("[malo [G]]")
	assert any("sin núcleo" in e for e in errores)


def test_grupo_no_admite_clausulas():
	assert not is_valid("[malo [G fuego [caliente fuego]]]")


def test_raiz_debe_ser_clausula():
	errores = validate("yo")
	assert any("raíz" in e for e in errores)


def test_corchete_sin_cerrar():
	with pytest.raises(NSMSyntaxError):
		parse("[ver yo algo")


def test_clausula_vacia():
	with pytest.raises(NSMSyntaxError):
		parse("[]")


def test_tokens_sobrantes():
	with pytest.raises(NSMSyntaxError):
		parse("[ver yo] algo")


def test_censo_opcional():
	vocab = {"yo", "algo"}
	assert is_valid("[ver yo algo]", vocab=vocab)
	errores = validate("[ver yo dragón]", vocab=vocab)
	assert any("fuera del censo" in e for e in errores)
