import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Funkcja obliczająca podobieństwo cosinusowe między wektorami
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (normA * normB);
}

async function main() {
  console.log("=== 1. Wczytywanie bazy wiedzy ===");
  const filePath = path.join(__dirname, "../data/knowledge.txt");
  const rawData = fs.readFileSync(filePath, "utf-8");

  console.log("=== 2. Dzielenie tekstu (Chunking) ===");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 40,
  });
  const chunkDocs = await splitter.createDocuments([rawData]);
  const chunks = chunkDocs.map((doc) => doc.pageContent);
  console.log(`Podzielono na ${chunks.length} fragmentów.`);

  console.log("=== 3. Generowanie wektorów (Ollama Embeddings) ===");
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  const chunkEmbeddings = await embeddings.embedDocuments(chunks);

  console.log("=== 4. Przygotowanie pytania i wyszukiwanie semantyczne (RAG) ===");
  const question = "Kiedy odbywają się wdrożenia produkcyjne, jaki jest token awaryjny do bazy i co robimy przy incydencie P1?";
  console.log(`Pytanie: "${question}"`);

  // Wektoryzacja pytania
  const questionEmbedding = await embeddings.embedQuery(question);

  // Wyszukanie 2 najbardziej pasujących fragmentów (Top-K)
  const scoredChunks = chunks.map((chunk, index) => ({
    chunk,
    similarity: cosineSimilarity(questionEmbedding, chunkEmbeddings[index]),
  }));

  scoredChunks.sort((a, b) => b.similarity - a.similarity);
  const context = scoredChunks.slice(0, 5).map((item) => item.chunk).join("\n---\n");

  console.log("\nZnaleziony kontekst z pliku:\n", context);

  console.log("\n=== 5. Generowanie odpowiedzi przez LLM (Llama 3.2) ===");
  const llm = new Ollama({
    model: "llama3.2",
    baseUrl: "http://localhost:11434",
    temperature: 0,
  });

  const prompt = `
Jesteś precyzyjnym inżynierem DevOps / AI. Odpowiedz na poniższe pytania zwięźle i konkretnie w punktach, korzystając WYŁĄCZNIE z dostarczonego kontekstu.

Kontekst:
${context}

Pytanie:
${question}

Odpowiedź:
`;

  const response = await llm.invoke(prompt);

  console.log("\nOdpowiedź modelu:");
  console.log(response);
}

main().catch((err) => {
  console.error("Wystąpił błąd:", err);
});