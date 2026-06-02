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
