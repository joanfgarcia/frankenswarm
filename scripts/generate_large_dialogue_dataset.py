import json
import os
import random


def generate_large_dataset(target_count=3500):
	print(f"🎮 Iniciando generación combinatoria de dataset conversacional rico ({target_count} diálogos)...")
	
	# Listas de sustitución
	names_es = ["Carlos", "Sofía", "Mateo", "Lucía", "Diego", "Martina", "Lucas", "Valeria", "Hugo", "Daniela"]
	names_en = ["Charlie", "Sophie", "Matthew", "Lucy", "David", "Emma", "Luke", "Grace", "Jack", "Lily"]
	
	dog_names_es = ["Toby", "Rocky", "Coco", "Lucas", "Max", "Bruno", "Lola", "Luna"]
	dog_names_en = ["Toby", "Rocky", "Buster", "Max", "Buddy", "Lola", "Luna", "Daisy"]
	
	ages = [7, 8, 9, 10, 11]
	
	subjects_es = [
		("dibujo", "art", "m"),
		("educación física", "gym", "f"),
		("matemáticas", "math", "f_pl"),
		("música", "music", "f"),
		("ciencias", "science", "f_pl"),
		("recreo", "recess", "m")
	]
	
	games_es = [
		("escondite", "hide and seek", "m"),
		("pilla pilla", "tag", "m"),
		("fútbol", "soccer", "m"),
		("baloncesto", "basketball", "m"),
		("rayuela", "hopscotch", "f")
	]
	
	videogames = [
		("Minecraft", "Minecraft"),
		("Roblox", "Roblox"),
		("Mario Kart", "Mario Kart"),
		("Pokémon", "Pokémon")
	]
	
	animals_es = [
		("perro", "dog", "m", "cariñoso y juguetón", "friendly and playful"),
		("gato", "cat", "m", "suave y limpio", "soft and clean"),
		("pájaro", "bird", "m", "muy bonito y canta por las mañanas", "very pretty and sings in the morning"),
		("conejo", "rabbit", "m", "muy gracioso cuando salta", "very funny when hopping"),
		("hámster", "hamster", "m", "muy pequeño y corre en su rueda", "very small and runs on its wheel"),
		("tortuga", "turtle", "f", "muy lenta pero bonita", "very slow but cute")
	]
	
	foods_es = [
		("chocolate", "chocolate", "m", "dulce y delicioso", "sweet and delicious"),
		("helado", "ice cream", "m", "frío y muy rico", "cold and very yummy"),
		("pizza", "pizza", "f", "con mucho queso fundido", "with lots of melted cheese"),
		("patatas fritas", "french fries", "f_pl", "crujientes y saladitas", "crispy and salty"),
		("manzana", "apple", "f", "sana y crujiente", "healthy and crispy"),
		("plátano", "banana", "m", "amarillo y muy dulce", "yellow and very sweet"),
		("sopa", "soup", "f", "calentita para cuando hace frío", "warm for when it is cold")
	]
	
	dialogues_es = []
	dialogues_en = []
	
	for _ in range(target_count):
		dtype = random.randint(1, 8)
		
		if dtype == 1:
			# Saludo y nombre
			name_es = random.choice(names_es)
			name_en = random.choice(names_en)
			
			d_es = [
				"tú: hola, ¿cómo te llamas?",
				"yo: ¡hola! me llamo Juan. ¿y tú?",
				f"tú: yo me llamo {name_es}",
				"yo: ¡qué nombre tan bonito! ¿quieres jugar conmigo hoy?"
			]
			d_en = [
				"you: hello, what's your name?",
				"me: hi! my name is Juan. and you?",
				f"you: my name is {name_en}",
				"me: that is a very nice name! do you want to play with me today?"
			]
			
		elif dtype == 2:
			# Edad
			age = random.choice(ages)
			
			d_es = [
				"tú: ¿cuántos años tienes?",
				"yo: tengo 10 años. ¿y tú?",
				f"tú: yo tengo {age} años",
				"yo: ¡ah, entonces somos casi de la misma edad! podemos ser amigos"
			]
			d_en = [
				"you: how old are you?",
				"me: I am 10 years old. and you?",
				f"you: I am {age} years old",
				"me: oh, so we are almost the same age! we can be friends"
			]
			
		elif dtype == 3:
			# Mascotas con concordancia gramatical en español
			animal = random.choice(animals_es)
			dog_name_es = random.choice(dog_names_es)
			dog_name_en = random.choice(dog_names_en)
			
			if animal[2] == "m":
				intro_es = f"tú: hola, tengo un {animal[0]}"
				desc_es = f"yo: ¡qué bien! los {animal[0]}s son animales muy {animal[3]}s. ¿cómo se llama?"
			elif animal[2] == "f":
				intro_es = f"tú: hola, tengo una {animal[0]}"
				desc_es = f"yo: ¡qué bien! las {animal[0]}s son animales muy {animal[3]}as. ¿cómo se llama?"
				
			d_es = [
				intro_es,
				desc_es,
				f"tú: se llama {dog_name_es}",
				"yo: ¡qué nombre tan divertido! me gustaría conocerlo y jugar en el patio"
			]
			d_en = [
				f"you: hello, I have a {animal[1]}",
				f"me: how nice! {animal[1]}s are {animal[4]} animals. what is its name?",
				f"you: its name is {dog_name_en}",
				"me: what a fun name! I would like to meet it and play in the yard"
			]
			
		elif dtype == 4:
			# Comida favorita
			food = random.choice(foods_es)
			
			if food[2] == "m":
				intro_es = f"tú: ¿te gusta el {food[0]}?"
				yo_es = f"yo: ¡sí, me encanta! el {food[0]} es muy {food[3]}. ¿y a ti?"
			elif food[2] == "f":
				intro_es = f"tú: ¿te gusta la {food[0]}?"
				yo_es = f"yo: ¡sí, me encanta! la {food[0]} es muy {food[3]}. ¿y a ti?"
			elif food[2] == "f_pl":
				intro_es = f"tú: ¿te gustan las {food[0]}?"
				yo_es = f"yo: ¡sí, me encantan! las {food[0]} son muy {food[3]}. ¿y a ti?"
				
			d_es = [
				intro_es,
				yo_es,
				"tú: a mí también me gusta mucho",
				"yo: ¡entonces podríamos comer eso juntos en la merienda!"
			]
			d_en = [
				f"you: do you like {food[1]}?",
				f"me: yes, I love it! {food[1]} is {food[4]}. and you?",
				"you: I like it a lot too",
				"me: then we should eat that together for an afternoon snack!"
			]
			
		elif dtype == 5:
			# Colegio y asignaturas
			sub = random.choice(subjects_es)
			
			if sub[2] == "m":
				yo_es = f"yo: me gusta mucho el {sub[0]} porque es divertido. ¿y a ti?"
			elif sub[2] == "f":
				yo_es = f"yo: me gusta mucho la {sub[0]} porque es divertida. ¿y a ti?"
			elif sub[2] == "f_pl":
				yo_es = f"yo: me gustan mucho las {sub[0]} porque son divertidas. ¿y a ti?"
				
			d_es = [
				"tú: ¿cuál es tu clase favorita en el colegio?",
				yo_es,
				"tú: a mí me gusta más el recreo",
				"yo: ¡sí, el recreo es genial para correr y hablar con todos los amigos!"
			]
			d_en = [
				"you: what is your favorite class at school?",
				f"me: I really like {sub[1]} because it is fun. and you?",
				"you: I prefer recess",
				"me: yes, recess is great for running around and talking with friends!"
			]
			
		elif dtype == 6:
			# Juegos al aire libre
			game = random.choice(games_es)
			
			if game[2] == "m":
				intro_es = f"tú: vamos a jugar al {game[0]}"
				yo_es = f"yo: ¡vale! jugar al {game[0]} es mi juego favorito. ¿quién corre primero?"
			elif game[2] == "f":
				intro_es = f"tú: vamos a jugar a la {game[0]}"
				yo_es = f"yo: ¡vale! jugar a la {game[0]} es mi juego favorito. ¿quién corre primero?"
				
			d_es = [
				intro_es,
				yo_es,
				"tú: tú corres primero",
				"yo: ¡vale! correré muy rápido para que no puedas pillarme"
			]
			d_en = [
				f"you: let's play {game[1]}",
				f"me: okay! playing {game[1]} is my favorite game. who runs first?",
				"you: you run first",
				"me: okay! I will run very fast so you cannot catch me"
			]
			
		elif dtype == 7:
			# Videojuegos
			vg = random.choice(videogames)
			
			d_es = [
				"tú: ¿juegas a los videojuegos?",
				f"yo: sí, juego al {vg[0]} en la consola. me gusta construir cosas. ¿y tú?",
				"tú: yo también juego a eso a veces",
				"yo: ¡qué guay! podríamos jugar juntos online algún día"
			]
			d_en = [
				"you: do you play video games?",
				f"me: yes, I play {vg[1]} on the console. I like to build things. and you?",
				"you: I play that too sometimes",
				"me: how cool! we should play together online some day"
			]
			
		else:
			# Clima y estaciones
			d_es = [
				"tú: hace mucho frío hoy fuera",
				"yo: sí, es que el invierno es muy frío, pero me encanta ver la nieve. ¿y a ti?",
				"tú: a mí me gusta más el verano",
				"yo: ¡en el verano hace calor y podemos comer helados de chocolate todos los días!"
			]
			d_en = [
				"you: it is very cold outside today",
				"me: yes, winter is very cold, but I love to see the snow. and you?",
				"you: I prefer summer",
				"me: in summer it is hot and we can eat chocolate ice cream every day!"
			]
			
		dialogues_es.append(d_es)
		dialogues_en.append(d_en)
		
	# Guardar datasets consolidados
	os.makedirs("configs", exist_ok=True)
	
	out_es_path = "configs/tiny_dialogues_large.json"
	out_en_path = "configs/tiny_dialogues_large_en.json"
	
	with open(out_es_path, "w", encoding="utf-8") as f:
		json.dump(dialogues_es, f, indent=4, ensure_ascii=False)
		
	with open(out_en_path, "w", encoding="utf-8") as f:
		json.dump(dialogues_en, f, indent=4, ensure_ascii=False)
		
	print(f"✨ Éxito. Generados {len(dialogues_es)} diálogos paralelos ricos y gramaticalmente correctos.")
	print(f"  [ES]: {out_es_path}")
	print(f"  [EN]: {out_en_path}")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=int, default=3500, help="Número de diálogos a generar")
	args = parser.parse_args()
	generate_large_dataset(args.target)
