import { tool } from "ai";
import { z } from "zod";
import { getAllEmbeddings } from "@/lib/db/queries";
import { getQueryEmbedding } from "@/lib/db/embedding-utils";

/**
 * Simple tokenizer - splits text into words (Chinese and English)
 */
function tokenize(text: string): string[] {
  // Remove punctuation and split by whitespace
  // For Chinese, we'll treat each character as a potential token
  // This is a simplified approach - in production, you might want to use a proper tokenizer
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

/**
 * BM25 algorithm implementation
 * @param documents - Array of document texts
 * @param query - Query text
 * @param k1 - Term frequency saturation parameter (default: 1.5)
 * @param b - Length normalization parameter (default: 0.75)
 * @returns Array of scores for each document
 */
function calculateBM25(
  documents: string[],
  query: string,
  k1 = 1.5,
  b = 0.75
): Array<{ index: number; score: number; text: string }> {
  const queryTerms = tokenize(query);
  const docTokens = documents.map((doc) => tokenize(doc));

  // Calculate average document length
  const avgDocLength =
    docTokens.reduce((sum, tokens) => sum + tokens.length, 0) /
    documents.length;

  // Calculate document frequencies (DF) for each query term
  const docFreq: Record<string, number> = {};
  for (const term of queryTerms) {
    docFreq[term] = docTokens.filter((tokens) =>
      tokens.includes(term)
    ).length;
  }

  // Calculate BM25 scores
  const scores: Array<{ index: number; score: number; text: string }> = [];

  for (let i = 0; i < documents.length; i++) {
    const docTokensList = docTokens[i];
    const docLength = docTokensList.length;
    let score = 0;

    // Count term frequencies in this document
    const termFreq: Record<string, number> = {};
    for (const token of docTokensList) {
      termFreq[token] = (termFreq[token] || 0) + 1;
    }

    // Calculate BM25 score for each query term
    for (const term of queryTerms) {
      const tf = termFreq[term] || 0;
      const df = docFreq[term] || 0;

      if (df === 0) continue;

      // Inverse document frequency (IDF)
      const idf = Math.log(
        (documents.length - df + 0.5) / (df + 0.5) + 1
      );

      // BM25 term score
      const numerator = tf * (k1 + 1);
      const denominator =
        tf + k1 * (1 - b + (b * docLength) / avgDocLength);
      const termScore = idf * (numerator / denominator);

      score += termScore;
    }

    scores.push({
      index: i,
      score,
      text: documents[i],
    });
  }

  return scores;
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;

  return dotProduct / denominator;
}

/**
 * Reciprocal Rank Fusion (RRF) algorithm
 * Combines multiple ranked lists into a single ranked list
 * @param rankedLists - Array of ranked lists, each containing { id, rank, score? }
 * @param k - RRF constant (default: 60)
 * @returns Fused ranked list
 */
function reciprocalRankFusion<T extends { id: string }>(
  rankedLists: Array<Array<{ id: string; rank: number; score?: number; data: T }>>,
  k = 60
): Array<{ id: string; rrfScore: number; data: T }> {
  const scoreMap = new Map<
    string,
    { rrfScore: number; data: T; ranks: number[] }
  >();

  // Calculate RRF score for each item
  for (const rankedList of rankedLists) {
    for (let rank = 0; rank < rankedList.length; rank++) {
      const item = rankedList[rank];
      const existing = scoreMap.get(item.id);

      if (existing) {
        existing.rrfScore += 1 / (k + rank + 1);
        existing.ranks.push(rank + 1);
      } else {
        scoreMap.set(item.id, {
          rrfScore: 1 / (k + rank + 1),
          data: item.data,
          ranks: [rank + 1],
        });
      }
    }
  }

  // Convert to array and sort by RRF score
  return Array.from(scoreMap.values())
    .map((item) => ({
      id: item.data.id,
      rrfScore: item.rrfScore,
      data: item.data,
    }))
    .sort((a, b) => b.rrfScore - a.rrfScore);
}

/**
 * Document search tool using hybrid retrieval (BM25 + Vector Similarity + RRF)
 * This tool searches through the embedding vector database to find relevant documents
 */
export const hybridSearchTool = tool({
  description:
    "Search for documents in the embedding vector database using hybrid retrieval. When users ask to search for documents, find information, or retrieve content, use this tool. It combines BM25 (keyword-based) and vector similarity search, then fuses results using Reciprocal Rank Fusion (RRF) for optimal accuracy. This provides the best of both keyword matching and semantic understanding.",
  inputSchema: z.object({
    query: z.string().describe("Search query text"),
    topK: z
      .number()
      .optional()
      .default(10)
      .describe("Number of top results to return (default: 10)"),
    bm25Weight: z
      .number()
      .optional()
      .default(1.0)
      .describe("Weight for BM25 results (default: 1.0)"),
    vectorWeight: z
      .number()
      .optional()
      .default(1.0)
      .describe("Weight for vector similarity results (default: 1.0)"),
    model: z
      .string()
      .optional()
      .default("text-embedding-v4")
      .describe("Embedding model to use for query vectorization"),
    dimension: z
      .number()
      .optional()
      .describe("Vector dimension (optional, uses model default if not specified)"),
  }),
  execute: async ({
    query,
    topK,
    bm25Weight,
    vectorWeight,
    model,
    dimension,
  }) => {
    try {
      if (!query || query.trim().length === 0) {
        return {
          success: false,
          error: "Query cannot be empty",
          results: [],
        };
      }

      // Get all embeddings from database
      const allEmbeddings = await getAllEmbeddings({ model });

      if (allEmbeddings.length === 0) {
        return {
          success: false,
          error: "No embeddings found in database. Please create embeddings first.",
          results: [],
        };
      }

      // Extract texts for BM25
      const texts = allEmbeddings.map((emb) => emb.text);

      // 1. BM25 Search
      const bm25Scores = calculateBM25(texts, query);
      const bm25Ranked = bm25Scores
        .filter((item) => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, topK * 2) // Get more candidates for RRF
        .map((item, rank) => ({
          id: allEmbeddings[item.index].id,
          rank: rank + 1,
          score: item.score * bm25Weight,
          data: allEmbeddings[item.index],
        }));

      // 2. Vector Similarity Search
      // Get query embedding
      const queryEmbeddingResult = await getQueryEmbedding({
        query,
        model,
        dimension,
      });

      if (!queryEmbeddingResult) {
        return {
          success: false,
          error: "Failed to generate embedding for query",
          results: [],
        };
      }

      // Calculate cosine similarities
      const vectorScores: Array<{
        index: number;
        score: number;
        embedding: typeof allEmbeddings[0];
      }> = [];

      for (let i = 0; i < allEmbeddings.length; i++) {
        const emb = allEmbeddings[i];
        const vector = emb.vector as number[];

        // Check if dimensions match
        if (vector.length !== queryEmbeddingResult.embedding.length) {
          continue; // Skip if dimensions don't match
        }

        const similarity = cosineSimilarity(
          queryEmbeddingResult.embedding,
          vector
        );

        vectorScores.push({
          index: i,
          score: similarity,
          embedding: emb,
        });
      }

      const vectorRanked = vectorScores
        .sort((a, b) => b.score - a.score)
        .slice(0, topK * 2) // Get more candidates for RRF
        .map((item, rank) => ({
          id: item.embedding.id,
          rank: rank + 1,
          score: item.score * vectorWeight,
          data: item.embedding,
        }));

      // 3. RRF Fusion
      const rankedLists = [];
      if (bm25Ranked.length > 0) {
        rankedLists.push(bm25Ranked);
      }
      if (vectorRanked.length > 0) {
        rankedLists.push(vectorRanked);
      }

      if (rankedLists.length === 0) {
        return {
          success: false,
          error: "No results found",
          results: [],
        };
      }

      const fusedResults = reciprocalRankFusion(rankedLists);

      // Return top K results
      const topResults = fusedResults.slice(0, topK).map((result, index) => ({
        rank: index + 1,
        id: result.id,
        text: result.data.text,
        rrfScore: result.rrfScore,
        metadata: result.data.metadata,
        model: result.data.model,
        dimension: result.data.dimension,
      }));

      return {
        success: true,
        message: `Found ${topResults.length} result(s)`,
        query,
        results: topResults,
        totalCandidates: {
          bm25: bm25Ranked.length,
          vector: vectorRanked.length,
          fused: fusedResults.length,
        },
      };
    } catch (error) {
      return {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to perform hybrid search. Please try again later.",
        results: [],
      };
    }
  },
});
