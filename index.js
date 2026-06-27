import express from "express";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import fs from "fs";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

dotenv.config();

const app = express();
app.use(express.json());

// llm model

const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 100,
  maxRetries: 2,
});

// ye vector embedding ke liye hai
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
  apiKey: process.env.GOOGLE_API_KEY,
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});

// yha par humari embedding ka part ho gya

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "grocery-store",
});

// parse pdf

const upload = async () => {
  // ye pdf se text ko extract karega
  const pdfPath = "./knowledge.pdf";
  const dataBuffer = fs.readFileSync(pdfPath);
  const pdfData = new PDFParse({ data: dataBuffer });
  const result = await pdfData.getText();
  const text = result.text;

  // ab text ko chunks me todenge via lanchain text splitter

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });

  const chunks = await splitter.createDocuments([text]);

  // ab vector db me data jo store karna hai
  // yha array me hi data aayega

  await vectorStore.addDocuments(chunks);
};

const PORT = process.env.PORT || 8080;

app.post("/ai", async (req, res) => {
  const { input } = req.body;
  // ab hum input me jo search denge uska search output dega vectore store se

  const docs = await vectorStore.similaritySearch(input, 5);

  const context = docs.map((d) => d.pageContent).join("/n");

  const response = await llm.invoke([
    new SystemMessage(`
You are a RAG AI Assistant.

STRICT RULES:
- Answer only from the provided context.
- Do not use outside knowledge.
- If the answer is not found in the context, reply exactly:
  "I don't know from uploaded PDF."

Context:
${context}
`),
    new HumanMessage(input),
  ]);

  res.status(200).json({ ai: response.content });
});

app.get("/", (req, res) => {
  res.status(200).send("server is working");
});

app.listen(PORT, () => {
  console.log(`server is running on port ${PORT}`);
});
