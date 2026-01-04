import "server-only";

/**
 * Get embedding vector for a query text using Alibaba Cloud DashScope API
 * This is used for search queries, not for storing embeddings
 */
export async function getQueryEmbedding({
  query,
  model = "text-embedding-v4",
  dimension,
}: {
  query: string;
  model?: string;
  dimension?: number;
}): Promise<{ embedding: number[]; dimension: number } | null> {
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
    input: { texts: [query] },
  };

  if (dimension) {
    requestBody.dimension = dimension;
  }

  try {
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

    if (!data.output?.embeddings?.[0]?.embedding) {
      return null;
    }

    const embedding = data.output.embeddings[0].embedding as number[];

    return {
      embedding,
      dimension: embedding.length,
    };
  } catch (error) {
    console.error("Failed to get query embedding:", error);
    return null;
  }
}
