% agent_name/2
agent_name('A', 'Nico') :- !.
agent_name('B', 'Sofy') :- !.
agent_name('C', 'Hugo') :- !.
agent_name('D', 'Domi') :- !.
agent_name(Id, Id).

% calcular_peso/6
% calcular_peso(MochilaComida, MochilaAgua, MochilaRamas, MochilaPiedras, TieneLanza, PesoMultiplicador)
calcular_peso(MochilaComida, MochilaAgua, MochilaRamas, MochilaPiedras, TieneLanza, PesoMultiplicador) :-
	PesoMultiplicador is 1.0 +
		0.10 * (MochilaComida + MochilaAgua) +
		0.15 * MochilaRamas +
		0.25 * MochilaPiedras +
		0.15 * TieneLanza.

% resolver_efecto_comida/8
resolver_efecto_comida(1, _, 60.0, 0.0, 0.0, Event, Source, 0) :-
	atom_concat('come comida cocinada (comida+) (', Source, Temp),
	atom_concat(Temp, ')', Event),
	!.
resolver_efecto_comida(0, RngValue, 40.0, -3.0, -15.0, Event, Source, 1) :-
	RngValue < 0.15,
	!,
	atom_concat('come comida cruda (', Source, Temp1),
	atom_concat(Temp1, ') | ¡intoxicación por comida cruda!', Event).
resolver_efecto_comida(0, _, 40.0, -3.0, 0.0, Event, Source, 0) :-
	atom_concat('come comida cruda (', Source, Temp),
	atom_concat(Temp, ')', Event).

% resolver_no_comida/6
resolver_no_comida(Skills, GroundFoodCap, _, _, AgentId, Event) :-
	GroundFoodCap >= 1.0,
	\+ member('comida', Skills),
	\+ member('pesca', Skills),
	!,
	agent_name(AgentId, Name),
	atom_concat(Name, ' no sabe recolectar comida del suelo', Event).
resolver_no_comida(_, _, _, _, _, 'no hay comida aquí ni en mochila').

% comer/18
% comer(FireActive, Skills, MochilaComida, MochilaAgua, GroundFoodCap, FoodEligible, FishEligible, RngValue, AgentId,
%       NewMochilaComida, NewMochilaAgua, NewGroundFoodCap, Success, AddHambre, AddSed, AddSalud, Event, Intoxicated)
comer(1, Skills, MochilaComida, MochilaAgua, GroundFoodCap, _, _, _, _,
      0, 0, GroundFoodCap, 1, 85.0, 50.0, 0.0, 'consume guiso caliente (comida++)', 0) :-
	MochilaComida > 0,
	MochilaAgua > 0,
	member('cocina', Skills),
	!.
comer(FireActive, Skills, MochilaComida, MochilaAgua, GroundFoodCap, FoodEligible, FishEligible, RngValue, _,
      MochilaComida, MochilaAgua, NewGroundFoodCap, 1, AddHambre, AddSed, AddSalud, Event, Intoxicated) :-
	GroundFoodCap >= 1.0,
	( (member('comida', Skills), FoodEligible =:= 1) ; (member('pesca', Skills), FishEligible =:= 1) ),
	!,
	NewGroundFoodCap is GroundFoodCap - 1.0,
	resolver_efecto_comida(FireActive, RngValue, AddHambre, AddSed, AddSalud, Event, 'suelo', Intoxicated).
comer(FireActive, _, MochilaComida, MochilaAgua, GroundFoodCap, _, _, RngValue, _,
      0, MochilaAgua, GroundFoodCap, 1, AddHambre, AddSed, AddSalud, Event, Intoxicated) :-
	MochilaComida > 0,
	!,
	resolver_efecto_comida(FireActive, RngValue, AddHambre, AddSed, AddSalud, Event, 'mochila', Intoxicated).
comer(_, Skills, 0, MochilaAgua, GroundFoodCap, FoodEligible, FishEligible, _, AgentId,
      0, MochilaAgua, GroundFoodCap, 0, 0.0, 0.0, 0.0, Event, 0) :-
	resolver_no_comida(Skills, GroundFoodCap, FoodEligible, FishEligible, AgentId, Event).

% fabricar/9
% fabricar(Skills, MochilaRamas, MochilaPiedras, TieneLanza, NewMochilaRamas, NewMochilaPiedras, NewTieneLanza, Success, Event)
fabricar(Skills, MochilaRamas, MochilaPiedras, TieneLanza, NewMochilaRamas, NewMochilaPiedras, NewTieneLanza, Success, Event) :-
	\+ member('artesanía', Skills),
	!,
	NewMochilaRamas = MochilaRamas,
	NewMochilaPiedras = MochilaPiedras,
	NewTieneLanza = TieneLanza,
	Success = 0,
	Event = 'no posee habilidad artesanía'.
fabricar(Skills, MochilaRamas, MochilaPiedras, TieneLanza, NewMochilaRamas, NewMochilaPiedras, NewTieneLanza, Success, Event) :-
	member('artesanía', Skills),
	(MochilaRamas < 1 ; MochilaPiedras < 1),
	!,
	NewMochilaRamas = MochilaRamas,
	NewMochilaPiedras = MochilaPiedras,
	NewTieneLanza = TieneLanza,
	Success = 0,
	Event = 'faltan materiales (necesita 1 rama y 1 piedra)'.
fabricar(Skills, MochilaRamas, MochilaPiedras, 1, NewMochilaRamas, NewMochilaPiedras, 1, Success, Event) :-
	member('artesanía', Skills),
	!,
	NewMochilaRamas = MochilaRamas,
	NewMochilaPiedras = MochilaPiedras,
	Success = 0,
	Event = 'ya tiene una lanza'.
fabricar(Skills, MochilaRamas, MochilaPiedras, 0, NewMochilaRamas, NewMochilaPiedras, 1, 1, 'fabrica una lanza') :-
	member('artesanía', Skills),
	MochilaRamas >= 1,
	MochilaPiedras >= 1,
	!,
	NewMochilaRamas is MochilaRamas - 1,
	NewMochilaPiedras is MochilaPiedras - 1.

% construir/10
% construir(Skills, MochilaRamas, MochilaPiedras, StormShelter, Location, NewMochilaRamas, NewMochilaPiedras, NewStormShelter, Success, Event)
construir(Skills, MochilaRamas, MochilaPiedras, StormShelter, _, NewMochilaRamas, NewMochilaPiedras, NewStormShelter, Success, Event) :-
	\+ member('construcción', Skills),
	!,
	NewMochilaRamas = MochilaRamas,
	NewMochilaPiedras = MochilaPiedras,
	NewStormShelter = StormShelter,
	Success = 0,
	Event = 'no posee habilidad construcción'.
construir(Skills, MochilaRamas, MochilaPiedras, StormShelter, _, NewMochilaRamas, NewMochilaPiedras, NewStormShelter, Success, Event) :-
	member('construcción', Skills),
	(MochilaRamas < 2 ; MochilaPiedras < 1),
	!,
	NewMochilaRamas = MochilaRamas,
	NewMochilaPiedras = MochilaPiedras,
	NewStormShelter = StormShelter,
	Success = 0,
	Event = 'faltan materiales (necesita 2 ramas y 1 piedra)'.
construir(Skills, MochilaRamas, MochilaPiedras, 1, _, NewMochilaRamas, NewMochilaPiedras, 1, Success, Event) :-
	member('construcción', Skills),
	!,
	NewMochilaRamas = MochilaRamas,
	NewMochilaPiedras = MochilaPiedras,
	Success = 0,
	Event = 'esta localización ya es un refugio'.
construir(Skills, MochilaRamas, MochilaPiedras, 0, Location, NewMochilaRamas, NewMochilaPiedras, 1, 1, Event) :-
	member('construcción', Skills),
	MochilaRamas >= 2,
	MochilaPiedras >= 1,
	!,
	NewMochilaRamas is MochilaRamas - 2,
	NewMochilaPiedras is MochilaPiedras - 1,
	atom_concat('construye un refugio permanente en ', Location, Event).

% encender/7
% encender(Skills, MochilaRamas, Location, NewMochilaRamas, FireDuration, Success, Event)
encender(Skills, MochilaRamas, _, NewMochilaRamas, FireDuration, Success, Event) :-
	\+ member('fuego', Skills),
	!,
	NewMochilaRamas = MochilaRamas,
	FireDuration = 0,
	Success = 0,
	Event = 'no posee habilidad fuego'.
encender(Skills, MochilaRamas, _, NewMochilaRamas, FireDuration, Success, Event) :-
	member('fuego', Skills),
	MochilaRamas < 2,
	!,
	NewMochilaRamas = MochilaRamas,
	FireDuration = 0,
	Success = 0,
	Event = 'faltan ramas (necesita 2 ramas)'.
encender(Skills, MochilaRamas, Location, NewMochilaRamas, 4, 1, Event) :-
	member('fuego', Skills),
	MochilaRamas >= 2,
	!,
	NewMochilaRamas is MochilaRamas - 2,
	atom_concat('enciende una hoguera en ', Location, Event).

% evaluar_depredador/7
% evaluar_depredador(DangerNearby, Action, TieneLanza, PredatorDamageMultiplier, NewTieneLanza, SubSalud, EventSuffix)
evaluar_depredador(1, Action, TieneLanza, PredatorDamageMultiplier, NewTieneLanza, SubSalud, EventSuffix) :-
	Action \= 'mover',
	Action \= 'luchar',
	Action \= 'dormir',
	!,
	( TieneLanza =:= 1 ->
		NewTieneLanza = 0,
		SubSalud = 0.0,
		EventSuffix = ' | depredador ataca pero es repelido con la lanza (se rompe)'
	;
		NewTieneLanza = 0,
		SubSalud is -20.0 * PredatorDamageMultiplier,
		EventSuffix = ' | depredador ataca'
	).
evaluar_depredador(_, _, TieneLanza, _, TieneLanza, 0.0, '').

% luchar/10
% luchar(DangerNearby, TieneLanza, RngValue, PredatorDamageMultiplier, NewTieneLanza, NewDangerNearby, Success, SubSalud, SubEnergia, Event)
luchar(0, TieneLanza, _, _, TieneLanza, 0, 0, 0.0, 0.0, 'no hay amenaza') :- !.
luchar(1, TieneLanza, RngValue, _, TieneLanza, 0, 1, 0.0, -12.0, 'lucha y gana') :-
	RngValue < 0.55, !.
luchar(1, 1, RngValue, _, 0, 0, 1, 0.0, -12.0, 'lucha y sobrevive usando la lanza (se rompe)') :-
	RngValue >= 0.55, !.
luchar(1, 0, RngValue, PredatorDamageMultiplier, 0, 1, 0, SubSalud, -12.0, 'lucha y pierde') :-
	RngValue >= 0.55, !,
	SubSalud is -30.0 * PredatorDamageMultiplier.

% enseñar_elegible/7
% enseñar_elegible(Location, TeacherSkills, StudentSkills, PreyLocation, FireActive, Success, Habilidad)
enseñar_elegible(Location, TeacherSkills, StudentSkills, _, _, 1, 'comida') :-
	(Location = 'bosque' ; Location = 'valle'),
	member('comida', TeacherSkills),
	\+ member('comida', StudentSkills),
	!.
enseñar_elegible(Location, TeacherSkills, StudentSkills, _, _, 1, 'agua') :-
	(Location = 'lago' ; Location = 'pantano'),
	member('agua', TeacherSkills),
	\+ member('agua', StudentSkills),
	!.
enseñar_elegible('río', TeacherSkills, StudentSkills, _, _, 1, 'comida') :-
	member('comida', TeacherSkills),
	\+ member('comida', StudentSkills),
	!.
enseñar_elegible('río', TeacherSkills, StudentSkills, _, _, 1, 'agua') :-
	member('agua', TeacherSkills),
	\+ member('agua', StudentSkills),
	!.
enseñar_elegible(Location, TeacherSkills, StudentSkills, PreyLocation, _, 1, 'caza') :-
	member('caza', TeacherSkills),
	\+ member('caza', StudentSkills),
	(Location = 'río' ; Location = PreyLocation),
	!.
enseñar_elegible(Location, TeacherSkills, StudentSkills, _, _, 1, 'artesanía') :-
	(Location = 'bosque' ; Location = 'montaña' ; Location = 'ruinas'),
	member('artesanía', TeacherSkills),
	\+ member('artesanía', StudentSkills),
	!.
enseñar_elegible(Location, TeacherSkills, StudentSkills, _, _, 1, 'construcción') :-
	(Location = 'cueva' ; Location = 'ruinas'),
	member('construcción', TeacherSkills),
	\+ member('construcción', StudentSkills),
	!.
enseñar_elegible(Location, TeacherSkills, StudentSkills, _, _, 1, 'pesca') :-
	(Location = 'río' ; Location = 'lago' ; Location = 'pantano'),
	member('pesca', TeacherSkills),
	\+ member('pesca', StudentSkills),
	!.
enseñar_elegible(Location, TeacherSkills, StudentSkills, _, _, 1, 'fuego') :-
	(Location = 'cueva' ; Location = 'bosque' ; Location = 'valle'),
	member('fuego', TeacherSkills),
	\+ member('fuego', StudentSkills),
	!.
enseñar_elegible(Location, TeacherSkills, StudentSkills, _, FireActive, 1, 'cocina') :-
	member('cocina', TeacherSkills),
	\+ member('cocina', StudentSkills),
	(Location = 'cueva' ; Location = 'valle' ; Location = 'río' ; FireActive =:= 1),
	!.
enseñar_elegible(_, _, _, _, _, 0, 'ninguno').

% ── Árbol Tecnológico ────────────────────────────────────────────────────────
prerrequisito('comida', []).
prerrequisito('agua', []).
prerrequisito('artesanía', []).
prerrequisito('caza', ['artesanía']).
prerrequisito('pesca', ['artesanía']).
prerrequisito('construcción', ['artesanía']).
prerrequisito('fuego', ['artesanía']).
prerrequisito('cocina', ['comida', 'agua', 'fuego']).

% Verifica si se cumplen los prerrequisitos
cumple_prereqs([], _) :- !.
cumple_prereqs([H|T], Skills) :-
	member(H, Skills),
	cumple_prereqs(T, Skills).

prerrequisitos_satisfechos(Habilidad, AgentSkills) :-
	prerrequisito(Habilidad, Prereqs),
	cumple_prereqs(Prereqs, AgentSkills).

% ── Consulta de Observación ─────────────────────────────────────────────────
% que_observar(AgentSkills, Location, FireActive, GroundFood, FoodEligible, FishEligible, GroundWater, WaterAvailable, PreyLocation, GroundBranches, BranchesEligible, GroundStones, StonesEligible, Habilidad)
que_observar(Skills, _, FireActive, _, _, _, _, _, _, _, _, _, _, 'fuego') :-
	\+ member('fuego', Skills),
	prerrequisitos_satisfechos('fuego', Skills),
	FireActive =:= 1.

que_observar(Skills, _, _, GroundFood, FoodEligible, _, _, _, _, _, _, _, _, 'comida') :-
	\+ member('comida', Skills),
	prerrequisitos_satisfechos('comida', Skills),
	GroundFood >= 1.0,
	FoodEligible =:= 1.

que_observar(Skills, _, _, GroundFood, _, FishEligible, _, _, _, _, _, _, _, 'pesca') :-
	\+ member('pesca', Skills),
	prerrequisitos_satisfechos('pesca', Skills),
	GroundFood >= 1.0,
	FishEligible =:= 1.

que_observar(Skills, _, _, _, _, _, GroundWater, WaterAvailable, _, _, _, _, _, 'agua') :-
	\+ member('agua', Skills),
	prerrequisitos_satisfechos('agua', Skills),
	GroundWater >= 1.0,
	WaterAvailable =:= 1.

que_observar(Skills, Location, _, _, _, _, _, _, PreyLocation, _, _, _, _, 'caza') :-
	\+ member('caza', Skills),
	prerrequisitos_satisfechos('caza', Skills),
	Location = PreyLocation.

que_observar(Skills, _, _, _, _, _, _, _, _, GroundBranches, BranchesEligible, GroundStones, StonesEligible, 'artesanía') :-
	\+ member('artesanía', Skills),
	prerrequisitos_satisfechos('artesanía', Skills),
	GroundBranches >= 1.0, BranchesEligible =:= 1,
	GroundStones >= 1.0, StonesEligible =:= 1.

que_observar(Skills, _, _, _, _, _, _, _, _, GroundBranches, BranchesEligible, GroundStones, StonesEligible, 'construcción') :-
	\+ member('construcción', Skills),
	prerrequisitos_satisfechos('construcción', Skills),
	GroundBranches >= 1.0, BranchesEligible =:= 1,
	GroundStones >= 1.0, StonesEligible =:= 1.

% ── Evaluación de Decisiones de Acción ──────────────────────────────────────
% evaluar_intento(Action, AgentSkills, MochilaComida, MochilaAgua, MochilaRamas, MochilaPiedras, TieneLanza, Location, GroundFood, GroundWater, GroundBranches, GroundStones, FireActive, PreyLocation, ResultType, Event)
% ResultType: 'exito', 'intento_valido', 'error_requisito', 'error_fisico', 'basal'

evaluar_intento('comer', Skills, MochilaComida, MochilaAgua, _, _, _, _, GroundFood, _, _, _, FireActive, _, ResultType, Event) :-
	( (MochilaComida > 0, MochilaAgua > 0, member('cocina', Skills), FireActive =:= 1) ->
		ResultType = 'exito', Event = 'cocina guiso caliente exitosamente'
	; (MochilaComida > 0) ->
		ResultType = 'exito', Event = 'come de la mochila exitosamente'
	; (GroundFood >= 1.0, (member('comida', Skills) ; member('pesca', Skills))) ->
		ResultType = 'exito', Event = 'come del suelo exitosamente'
	; (GroundFood >= 1.0, \+ member('comida', Skills), \+ member('pesca', Skills)) ->
		( prerrequisitos_satisfechos('comida', Skills) ->
			ResultType = 'intento_valido', Event = 'intenta comer/recolectar comida para aprender'
		;
			ResultType = 'error_requisito', Event = 'intenta comer sin prerrequisito'
		)
	;
		ResultType = 'error_fisico', Event = 'intenta comer de suelo agotado'
	),
	!.

evaluar_intento('beber', Skills, _, MochilaAgua, _, _, _, _, _, GroundWater, _, _, _, _, ResultType, Event) :-
	( (MochilaAgua > 0) ->
		ResultType = 'exito', Event = 'bebe de la mochila exitosamente'
	; (GroundWater >= 1.0, member('agua', Skills)) ->
		ResultType = 'exito', Event = 'bebe del suelo exitosamente'
	; (GroundWater >= 1.0, \+ member('agua', Skills)) ->
		( prerrequisitos_satisfechos('agua', Skills) ->
			ResultType = 'intento_valido', Event = 'intenta beber para aprender'
		;
			ResultType = 'error_requisito', Event = 'intenta beber sin prerrequisito'
		)
	;
		ResultType = 'error_fisico', Event = 'intenta beber de zona seca'
	),
	!.

evaluar_intento('fabricar', Skills, _, _, MochilaRamas, MochilaPiedras, TieneLanza, _, _, _, _, _, _, _, ResultType, Event) :-
	( (\+ member('artesanía', Skills)) ->
		( prerrequisitos_satisfechos('artesanía', Skills) ->
			( (MochilaRamas >= 1, MochilaPiedras >= 1) ->
				ResultType = 'intento_valido', Event = 'intenta fabricar para aprender artesania'
			;
				ResultType = 'error_fisico', Event = 'intenta fabricar pero no tiene materiales'
			)
		;
			ResultType = 'error_requisito', Event = 'intenta fabricar sin prerrequisito'
		)
	; (TieneLanza =:= 1) ->
		ResultType = 'error_fisico', Event = 'ya tiene una lanza'
	; (MochilaRamas < 1 ; MochilaPiedras < 1) ->
		ResultType = 'error_fisico', Event = 'faltan materiales para fabricar'
	;
		ResultType = 'exito', Event = 'fabrica una lanza exitosamente'
	),
	!.

evaluar_intento('construir', Skills, _, _, MochilaRamas, MochilaPiedras, _, _, _, _, _, _, _, _, ResultType, Event) :-
	( (\+ member('construcción', Skills)) ->
		( prerrequisitos_satisfechos('construcción', Skills) ->
			( (MochilaRamas >= 2, MochilaPiedras >= 1) ->
				ResultType = 'intento_valido', Event = 'intenta construir para aprender construccion'
			;
				ResultType = 'error_fisico', Event = 'intenta construir pero no tiene materiales'
			)
		;
			ResultType = 'error_requisito', Event = 'intenta construir sin prerrequisito'
		)
	; (MochilaRamas < 2 ; MochilaPiedras < 1) ->
		ResultType = 'error_fisico', Event = 'faltan materiales para construir'
	;
		ResultType = 'exito', Event = 'construye un refugio exitosamente'
	),
	!.

evaluar_intento('encender', Skills, _, _, MochilaRamas, _, _, _, _, _, _, _, FireActive, _, ResultType, Event) :-
	( (\+ member('fuego', Skills)) ->
		( prerrequisitos_satisfechos('fuego', Skills) ->
			( (MochilaRamas >= 2) ->
				ResultType = 'intento_valido', Event = 'intenta encender para aprender fuego'
			;
				ResultType = 'error_fisico', Event = 'intenta encender pero no tiene ramas'
			)
		;
			ResultType = 'error_requisito', Event = 'intenta encender sin prerrequisito'
		)
	; (FireActive =:= 1) ->
		ResultType = 'error_fisico', Event = 'ya hay una hoguera activa'
	; (MochilaRamas < 2) ->
		ResultType = 'error_fisico', Event = 'faltan ramas para encender'
	;
		ResultType = 'exito', Event = 'enciende una hoguera exitosamente'
	),
	!.

evaluar_intento(_, _, _, _, _, _, _, _, _, _, _, _, _, _, 'basal', 'accion basal o comunicativa').

