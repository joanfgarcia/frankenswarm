const fs = require('fs');

const text = fs.readFileSync('/home/joan/.gemini/antigravity/brain/986706db-9916-42f4-9fef-59f7b69db4be/.system_generated/tasks/task-4232.log', 'utf8');

function parseLog(text) {
	const lines = text.split('\n');
	
	let isSimulation = text.includes("[Tick ") || text.includes("SIMULACIÓN EN LA ARENA");

	if (isSimulation) {
		return parseSimulationLog(lines);
	} else {
		return parseTrainingLog(lines);
	}
}

function parseSimulationLog(lines) {
	return { logType: 'simulation', ticks: [] };
}

function parseTrainingLog(lines) {
	const cleanedLines = [];
	let buffer = "";

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i].trim();
		if (!line) continue;

		if (line.startsWith("Ep")) {
			if (buffer) {
				cleanedLines.push(buffer);
			}
			buffer = line;
		} else if (line.startsWith("🧬") || line.startsWith("💤") || line.startsWith("🏆") || line.startsWith("Maestro") || line.startsWith("Running") || line.startsWith("Loaded") || line.startsWith("🔒") || line.includes("Arena") || line.includes("COMUNICACIÓN")) {
			if (buffer) {
				cleanedLines.push(buffer);
				buffer = "";
			}
			cleanedLines.push(line);
		} else {
			if (buffer) {
				const lastChar = buffer.slice(-1);
				const firstChar = line.slice(0, 1);
				const isWordOrNum = /[\wáéíóúñÁÉÍÓÚÑ\.:-]/;
				if (isWordOrNum.test(lastChar) && isWordOrNum.test(firstChar)) {
					buffer += line;
				} else {
					buffer += " " + line;
				}
			} else {
				cleanedLines.push(line);
			}
		}
	}
	if (buffer) {
		cleanedLines.push(buffer);
	}

	const data = {
		logType: 'training',
		episodes: [],
		survivalA: [],
		survivalB: [],
		survivalC: [],
		rawLines: cleanedLines
	};

	let currentEpisode = 0;

	for (let i = 0; i < cleanedLines.length; i++) {
		const line = cleanedLines[i];

		const epNumMatch = line.match(/Ep\s+(\d+)/);

		if (epNumMatch) {
			const ep = parseInt(epNumMatch[1]);
			currentEpisode = ep;

			const getVal = (regex, defaultVal = null) => {
				const m = line.match(regex);
				return m ? m[1] : defaultVal;
			};

			const sA = getVal(/A:\s*(\d+|--)/);
			const sB = getVal(/B:\s*(\d+|--)/);
			const sC = getVal(/C:\s*(\d+|--)/);

			data.episodes.push(ep);
			data.survivalA.push(sA === '--' || sA === null ? null : parseInt(sA));
			data.survivalB.push(sB === '--' || sB === null ? null : parseInt(sB));
			data.survivalC.push(sC === '--' || sC === null ? null : parseInt(sC));
		}
	}
	return data;
}

const result = parseLog(text);
console.log("LogType detected:", result.logType);
console.log("Episodes found:", result.episodes.length);
if (result.episodes.length > 0) {
	console.log("First 3 episodes:", result.episodes.slice(0, 3));
	console.log("Last 3 episodes:", result.episodes.slice(-3));
} else {
	console.log("Cleaned lines sample:", result.rawLines.slice(0, 30));
}
