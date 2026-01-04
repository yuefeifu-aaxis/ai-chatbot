import { tool } from "ai";
import { z } from "zod";
import { saveEmbeddings } from "@/lib/db/queries";

/**
 * Get embedding vectors from Alibaba Cloud DashScope API
 * @param texts - Array of texts to embed (up to 10 per request)
 * @param model - Model name (default: text-embedding-v4)
 * @param dimension - Vector dimension (optional)
 * @returns Array of embedding vectors with their dimensions
 */
async function getEmbeddingsFromAPI({
  texts,
  model = "text-embedding-v4",
  dimension,
}: {
  texts: string[];
  model?: string;
  dimension?: number;
}) {
  const apiKey = process.env.DASHSCOPE_API_KEY;
  if (!apiKey) {
    throw new Error("DASHSCOPE_API_KEY environment variable is not set");
  }

  const baseUrl =
    process.env.DASHSCOPE_BASE_URL ||
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding";

  const requestBody: {
    model: string;
    input: { texts: string[] };
    dimension?: number;
  } = {
    model,
    input: { texts },
  };

  if (dimension) {
    requestBody.dimension = dimension;
  }

  const response = await fetch(baseUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `DashScope API error: ${response.status} ${response.statusText} - ${errorText}`
    );
  }

  const data = await response.json();

  if (!data.output?.embeddings || data.output.embeddings.length === 0) {
    throw new Error("Invalid response from DashScope API");
  }

  // Map embeddings using index to ensure correct order
  return data.output.embeddings.map(
    (emb: { index?: number; embedding: number[] }) => ({
      index: emb.index ?? 0,
      embedding: emb.embedding as number[],
      dimension: emb.embedding.length,
    })
  );
}

/**
 * Embedding tool for converting text chunks to vectors using Alibaba Cloud DashScope API
 * and saving them to the database
 */
export const embeddingTool = tool({
  description:
    "Convert text chunks to embedding vectors using Alibaba Cloud DashScope API and save them to the database. Use this tool after splitting text into chunks to create vector representations for semantic search and retrieval.",
  inputSchema: z.object({
    chunks: z
      .array(z.string())
      .describe("Array of text chunks to convert to embeddings"),
    model: z
      .string()
      .optional()
      .default("text-embedding-v4")
      .describe("Embedding model name (default: text-embedding-v4)"),
    dimension: z
      .number()
      .optional()
      .describe(
        "Vector dimension (optional, model default will be used if not specified)"
      ),
    metadata: z
      .record(z.any())
      .optional()
      .describe("Optional metadata to store with embeddings"),
  }),
  execute: async ({ chunks, model, dimension, metadata }) => {
    try {
      if (!chunks || chunks.length === 0) {
        return {
          success: false,
          error: "No chunks provided for embedding",
          savedCount: 0,
        };
      }

      const embeddings = [];
      const errors = [];

      // Process chunks in batches (DashScope supports up to 10 texts per request)
      const batchSize = 10;
      for (let i = 0; i < chunks.length; i += batchSize) {
        const batch = chunks.slice(i, i + batchSize);

        try {
          // Process batch of texts together
          const embeddingResults = await getEmbeddingsFromAPI({
            texts: batch,
            model,
            dimension,
          });

          // Map embeddings back to their original texts using index
          // Sort by index to ensure correct order
          const sortedResults = [...embeddingResults].sort(
            (a, b) => a.index - b.index
          );

          for (let j = 0; j < batch.length; j++) {
            const result = sortedResults.find((r) => r.index === j) || sortedResults[j];
            if (result && result.embedding) {
              embeddings.push({
                text: batch[j],
                vector: result.embedding,
                model,
                dimension: result.dimension.toString(),
                metadata: metadata || null,
              });
            } else {
              errors.push({
                chunk: batch[j].substring(0, 50) + "...",
                error: "No embedding returned for this chunk",
              });
            }
          }
        } catch (error) {
          // If batch fails, try processing individually
          for (const chunk of batch) {
            try {
              const results = await getEmbeddingsFromAPI({
                texts: [chunk],
                model,
                dimension,
              });

              const result = results[0];
              if (result && result.embedding) {
                embeddings.push({
                  text: chunk,
                  vector: result.embedding,
                  model,
                  dimension: result.dimension.toString(),
                  metadata: metadata || null,
                });
              } else {
                errors.push({
                  chunk: chunk.substring(0, 50) + "...",
                  error: "No embedding returned for this chunk",
                });
              }
            } catch (individualError) {
              errors.push({
                chunk: chunk.substring(0, 50) + "...",
                error:
                  individualError instanceof Error
                    ? individualError.message
                    : "Unknown error",
              });
            }
          }
        }
      }

      if (embeddings.length === 0) {
        return {
          success: false,
          error: "Failed to generate any embeddings",
          errors,
          savedCount: 0,
        };
      }

      // Save embeddings to database
      const savedEmbeddings = await saveEmbeddings({ embeddings });

      return {
        success: true,
        message: `Successfully generated and saved ${savedEmbeddings.length} embedding(s)`,
        savedCount: savedEmbeddings.length,
        totalChunks: chunks.length,
        model,
        dimension: embeddings[0]?.dimension,
        errors: errors.length > 0 ? errors : undefined,
      };
    } catch (error) {
      return {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to generate embeddings. Please try again later.",
        savedCount: 0,
      };
    }
  },
});
