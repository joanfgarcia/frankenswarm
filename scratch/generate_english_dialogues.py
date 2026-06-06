import glob
import json

translation_map = {
	"yo: qué juego desea jugar hoy": "me: what game do you want to play today",
	"tú: vamos a jugar al mosca": "you: let's play tag",
	"yo: al mosca qué equipo seremos": "me: in tag which team will we be",
	"tú: seremos los rojo": "you: we will be the red ones",
	"yo: qué tiempo hace desde que no fuimos al parque": "me: how long has it been since we went to the park",
	"tú: fue hace una semana": "you: it was a week ago",
	"yo: una semana qué cosas jugar allí": "me: one week what things did you play there",
	"tú: haría a la cálmate": "you: I would go on the swing",
	"yo: te gusta jugar a la ciruela": "me: do you like to play plum",
	"tú: sí te gusta": "you: yes I like it",
	"yo: te gusta jugar a la ciruela qué color seré": "me: you like to play plum what color will I be",
	"tú: seré amarillo": "you: I will be yellow",
	"tú: vamos a jugar a la caza al gato": "you: let's play catch the cat",
	"yo: a la caza al gato qué equipo seremos": "me: in catch the cat which team will we be",
	"tú: seremos los azul": "you: we will be the blue ones",
	"yo: qué cosas tienes en casa": "me: what things do you have at home",
	"tú: tengo una pelota una barro y un avión": "you: I have a ball some clay and a toy plane",
	"yo: un avión qué tipo de avión": "me: a plane what type of plane",
	"tú: un avión espacio": "you: a space plane",
	"tú: vamos a jugar al fútbol sala": "you: let's play futsal",
	"yo: al fútbol sala qué equipo seremos": "me: in futsal which team will we be",
	"tú: seremos los blancos": "you: we will be the white ones",
	"yo: te gusta jugar al pan": "me: do you like to play bread",
	"yo: te gusta jugar al pan qué color seré": "me: you like to play bread what color will I be",
	"tú: seré negro": "you: I will be black",
	"tú: fue hace un mes": "you: it was a month ago",
	"yo: un mes qué cosas haría allí": "me: one month what things would you do there",
	"tú: haría al barro": "you: I would play with mud",
	"tú: vamos a jugar al duele de mesa": "you: let's play board games",
	"yo: al duele de mesa qué equipo seremos": "me: in board games which team will we be",
	"tú: seremos los verde": "you: we will be the green ones",
	"tú: seré rojo": "you: I will be red",
	"yo: qué tiempo hace anteriormente que no fuimos al parque": "me: how long has it been since we went to the park",
	"tú: fue hace un día": "you: it was a day ago",
	"yo: un día qué cosas haría allí": "me: one day what things would you do there",
	"tú: vamos a jugar a la pista": "you: let's play track",
	"yo: a la pista qué equipo seremos": "me: in track which team will we be",
	"yo: te gusta jugar a la bruja qué color seré": "me: you like to play witch what color will I be",
	"tú: hola tengo un gato": "you: hello I have a cat",
	"yo: el gato es muy suave y tú": "me: the cat is very soft and you",
	"tú: no tengo un perro": "you: no I have a dog",
	"yo: el perro es muy lío es muy felices de verte": "me: the dog is very messy it is very happy to see you",
	"tú: veo a un pedazo de color": "you: I see a colored bird",
	"yo: es un pájaro color y tú qué pájaro vemos": "me: it is a colored bird and you what bird do we see",
	"tú: no se me acuerdas qué es su nombre": "you: I don't remember its name",
	"yo: es un pájaro pájaro": "me: it is a bird bird",
	"tú: mi animal favorito es el gato": "you: my favorite animal is the cat",
	"yo: los gato son muy encantado me encanta jugar contigo conmigo": "me: cats are very lovely I love to play with you",
	"tú: y tú": "you: and you",
	"yo: mi animal favorito es el perro me encanta jugar contigo y correr juntos": "me: my favorite animal is the dog I love to play with you and run together",
	"tú: mi animal hogar favorito es mi perro es muy inteligente": "you: my favorite pet is my dog it is very smart",
	"yo: los perros son muy inteligente me encanta jugar contigo": "me: dogs are very smart I love to play with you",
	"yo: mi animal hogar favorito es mi gato me encanta jugar contigo y dormir juntos": "me: my favorite pet is my cat I love to play with you and sleep together",
	"tú: quiero tener un gato": "you: I want to have a cat",
	"yo: los gato son muy encantado serán felices de estén conmigo": "me: cats are very lovely they will be happy to be with me",
	"yo: quiero tener un perro serás mi mejor amigo": "me: I want to have a dog you will be my best friend",
	"tú: el pájaro que viste un sombrero se llama un pájaro jodido": "you: the bird that wears a hat is called a silly bird",
	"yo: es correcto el pájaro jodido viste un sombrero para proteger del sol": "me: that's right the silly bird wears a hat to protect from the sun",
	"tú: los cálmate y las echar costa viven en jardín": "you: the snails and the lizards live in the garden",
	"yo: es correcto los cálmate y las echar costa se pueden ver en jardín y sus noche": "me: that's right the snails and the lizards can be seen in the garden at night",
	"tú: el gato come ajo": "you: the cat eats garlic",
	"yo: es equivocada los gato no tenido ajo son peligroso para su salud": "me: that is wrong cats should not have garlic it is dangerous for their health",
	"tú: los pájaro tienen sepas": "you: the birds have wings",
	"yo: es correcto los pájaro tienen sepas para volar": "me: that's right the birds have wings to fly",
	"tú: el gato se seas ver en las montaña": "you: the cat can be seen in the mountains",
	"yo: es equivocada los gato no se seas ver en las montaña pero los perros sí": "me: that is wrong cats are not seen in the mountains but dogs are",
	"tú: los perros y los gato tienen pelo suave": "you: dogs and cats have soft fur",
	"yo: es correcto los perros y los gato tienen pelo suave y se les conocen por el cálmate": "me: that's right dogs and cats have soft fur and are known to be calm",
	"yo: qué comida tienes hoy": "me: what food do you have today",
	"tú: tengo una manzana y un huevo comido": "you: I have an apple and a boiled egg",
	"yo: qué comido parece": "me: what food it seems",
	"yo: qué tipo de comida te gusta más": "me: what type of food do you like most",
	"tú: me encanta el truco": "you: I love candy",
	"yo: me encanta también el truco es muy dulce y hermoso": "me: I also love candy it is very sweet and beautiful",
	"yo: has intentado alguna fruta hoy": "me: have you tried any fruit today",
	"tú: he comido un plátano": "you: I ate a banana",
	"yo: el plátano es una fruta muy comido": "me: the banana is a very eaten fruit",
	"yo: qué color tiene tu manzana": "me: what color is your apple",
	"tú: es roja": "you: it is red",
	"yo: es hermosa y brillante": "me: it is beautiful and bright",
	"yo: qué tipo de comida vamos a comer en la fiesta": "me: what type of food are we going to eat at the party",
	"tú: vamos a comer una tibio": "you: we are going to eat a warm soup",
	"yo: es un comida muy rápido de preparado y muy maravilloso": "me: it is a very fast food to prepare and very wonderful",
	"yo: has intentado una manzana en queso": "me: have you tried apple with cheese",
	"tú: no qué pasa si la probar": "you: no what happens if I try it",
	"yo: queda bien interesante qué piensas": "me: it is quite interesting what do you think",
	"yo: qué fruta te gusta más en el invierno": "me: what fruit do you like most in the winter",
	"tú: a mi me encanta el asustada saciado": "you: I love baked pears",
	"yo: también es muy dulce y cuanto en comida": "me: it is also very sweet and good food",
	"yo: qué tipo de comida tienes en tu cumpleaños": "me: what type of food do you have on your birthday",
	"tú: tienes un pastel de chocolate": "you: you have a chocolate cake",
	"yo: es un pastel maravilloso y rico": "me: it is a wonderful and delicious cake",
	"yo: has intentado una fruta en forma de jugar": "me: have you tried fruit shaped like toys",
	"yo: queda muy interesante es una fruta saciado": "me: it is very interesting it is a good fruit",
	"yo: qué color tiene un plátano edad": "me: what color is a ripe banana",
	"tú: es negro": "you: it is black",
	"yo: es un poco oscuro pero es maravilloso y edad": "me: it is a bit dark but it is wonderful and ripe",
	"yo: qué tipo de comida tienes en tu fiesta": "me: what type of food do you have at your party",
	"tú: vamos a comer un comida con pollo": "you: we are going to eat chicken food",
	"yo: también es un comida muy maravilloso y lleno de sabor": "me: it is also a very wonderful food full of flavor",
	"yo: has intentado una manzana en queso en una almuerzo": "me: have you tried apple with cheese for lunch",
	"yo: queda muy interesante me gusta mucho": "me: it is very interesting I like it a lot",
	"yo: qué fruta te gusta más en verano": "me: what fruit do you like most in summer",
	"tú: a mi me encanta el agua alegre": "you: I love watermelon",
	"yo: es una fruta cuanto en agua y comida": "me: it is a fruit rich in water and food",
	"tú: tienes una manzana al humo": "you: you have a baked apple",
	"yo: es un comida muy pasé y suave": "me: it is a very soft and easy food",
	"yo: has intentado una fruta en forma de helado": "me: have you tried fruit shaped like ice cream",
	"yo: queda muy interesante es una fruta helado": "me: it is very interesting it is an ice cream fruit",
	"yo: qué color tiene una plátano verde": "me: what color is a green banana",
	"tú: es verde": "you: it is green",
	"yo: es un poco clara pero es alegre y cuanto en comida": "me: it is a bit light but it is fresh and good food",
	"tú: vamos a comer una queso fin": "you: we are going to eat fine cheese",
	"yo: también es un comida muy maravilloso y hermoso": "me: it is also a very wonderful and beautiful food",
	"yo: has intentado una manzana en queso en un diste": "me: have you tried apple with cheese for dessert",
	"yo: qué hayas contigo ahora": "me: what do you have with you now",
	"tú: jugar al aire él": "you: playing outside",
	"yo: cuántos golpe carter": "me: how many hits do you have",
	"yo: qué haces contigo": "me: what are you doing",
	"tú: estoy leído un libro": "you: I am reading a book",
	"yo: qué pasa en la historia": "me: what happens in the story",
	"yo: hola mamá": "me: hello mom",
	"tú: hola hermano": "you: hello brother",
	"yo: qué tal estás": "me: how are you",
	"yo: qué quieres comer": "me: what do you want to eat",
	"tú: quiero pastel de queso": "you: I want cheesecake",
	"yo: queremos pedir pizza": "me: we want to order pizza",
	"yo: qué pasa en casa": "me: what is happening at home",
	"tú: estamos viendo una película": "you: we are watching a movie",
	"yo: qué película": "me: what movie",
	"yo: qué hace contigo ahora": "me: what are you doing now",
	"tú: estoy pintura un gato": "you: I am painting a cat",
	"yo: qué color tenías": "me: what color was it",
	"yo: qué lee en ese libro": "me: what are you reading in that book",
	"tú: está leído una historia de seas": "you: I am reading a history of oceans",
	"yo: hola papá": "me: hello dad",
	"tú: quiero pollo pienso": "you: I want roast chicken",
	"yo: queremos pedir taza": "me: we want to order soup",
	"tú: estamos viendo una partido de fútbol": "you: we are watching a football game",
	"yo: qué equipo pasaba": "me: what team was winning",
	"tú: estoy construir una torre de partes": "you: I am building a tower of blocks",
	"yo: cuántos piso tiene": "me: how many floors does it have",
	"tú: está leído una historia de héroe": "you: I am reading a story of heroes",
	"tú: quiero carne": "you: I want meat",
	"tú: estamos bailar queso": "you: we are dancing music",
	"yo: qué canción bailar": "me: what song are you dancing",
	"tú: estoy jugando con los dedos": "you: I am playing with my fingers",
	"yo: qué juego estás jugando": "me: what game are you playing"
}

def translate_dialogues():
	files = glob.glob("configs/tiny_dialogues*.json")
	for fpath in files:
		if "_en" in fpath:
			continue
		with open(fpath, encoding="utf-8") as f:
			data = json.load(f)
		
		translated_data = []
		for dialogue in data:
			translated_dialogue = []
			for turn in dialogue:
				# Buscar traducción exacta o por turnos
				clean_turn = turn.strip()
				if clean_turn in translation_map:
					translated_dialogue.append(translation_map[clean_turn])
				else:
					# Fallback por si acaso, traduciendo tags
					translated = clean_turn.replace("yo: ", "me: ").replace("tú: ", "you: ")
					# Reemplazos básicos genéricos
					translated = translated.replace("gato", "cat").replace("perro", "dog").replace("manzana", "apple")
					translated_dialogue.append(translated)
			translated_data.append(translated_dialogue)
			
		out_path = fpath.replace(".json", "_en.json")
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(translated_data, f, ensure_ascii=False, indent=4)
		print(f"Traducido {fpath} -> {out_path}")

if __name__ == "__main__":
	translate_dialogues()
