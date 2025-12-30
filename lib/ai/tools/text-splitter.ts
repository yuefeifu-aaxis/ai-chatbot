import { tool } from "ai";
import { z } from "zod";

/**
 * RecursiveCharacterTextSplitter implementation
 * Splits text recursively by trying different separators in order:
 * 1. Double newlines (paragraphs)
 * 2. Single newlines
 * 3. Sentence endings (. ! ?)
 * 4. Spaces
 * 5. Characters
 */
class RecursiveCharacterTextSplitter {
  private chunkSize: number;
  private chunkOverlap: number;
  private separators: string[];

  constructor({
    chunkSize = 1000,
    chunkOverlap = 200,
  }: {
    chunkSize?: number;
    chunkOverlap?: number;
  } = {}) {
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    // Separators in order of preference (from largest to smallest units)
    this.separators = [
      "\n\n", // Paragraphs
      "\n", // Lines
      ". ", // Sentences (with space after)
      "! ", // Exclamations
      "? ", // Questions
      " ", // Words
      "", // Characters (fallback)
    ];
  }

  /**
   * Split text into chunks
   */
  splitText(text: string): string[] {
    const chunks: string[] = [];
    this.splitTextRecursive(text, chunks);
    return chunks.filter((chunk) => chunk.trim().length > 0);
  }

  /**
   * Recursively split text using separators
   */
  private splitTextRecursive(text: string, chunks: string[]): void {
    // If text is small enough, add it directly
    if (text.length <= this.chunkSize) {
      if (text.trim()) {
        chunks.push(text.trim());
      }
      return;
    }

    // Try each separator in order
    for (const separator of this.separators) {
      if (separator === "") {
        // Last resort: split by character
        this.splitByCharacter(text, chunks);
        return;
      }

      const splits = this.splitBySeparator(text, separator);
      
      // If we got multiple splits, process them
      if (splits.length > 1) {
        const processedSplits: string[] = [];
        
        for (const split of splits) {
          if (split.length <= this.chunkSize) {
            // Small enough, keep it
            if (split.trim()) {
              processedSplits.push(split.trim());
            }
          } else {
            // Too large, recurse with smaller separators
            const subChunks: string[] = [];
            this.splitTextRecursive(split, subChunks);
            processedSplits.push(...subChunks);
          }
        }
        
        // Merge processed splits with overlap
        if (processedSplits.length > 0) {
          this.mergeSplitsWithOverlap(processedSplits, separator, chunks);
        }
        
        return;
      }
    }
    
    // If no separator worked, just split by character
    this.splitByCharacter(text, chunks);
  }

  /**
   * Split text by a specific separator
   */
  private splitBySeparator(text: string, separator: string): string[] {
    if (separator === "") {
      return [text];
    }
    return text.split(separator);
  }

  /**
   * Merge splits with overlap
   */
  private mergeSplitsWithOverlap(
    splits: string[],
    separator: string,
    chunks: string[]
  ): void {
    let currentChunk = "";
    
    for (let i = 0; i < splits.length; i++) {
      const split = splits[i];
      const separatorToAdd = separator === " " ? " " : separator;
      
      // Calculate size if we add this split
      const sizeIfAdded =
        currentChunk.length +
        (currentChunk ? separatorToAdd.length : 0) +
        split.length;
      
      // If adding this split would exceed chunk size
      if (sizeIfAdded > this.chunkSize && currentChunk) {
        // Save current chunk
        chunks.push(currentChunk.trim());
        
        // Start new chunk with overlap
        if (
          this.chunkOverlap > 0 &&
          currentChunk.length > this.chunkOverlap
        ) {
          const overlapText = currentChunk.slice(-this.chunkOverlap);
          currentChunk = overlapText + separatorToAdd + split;
        } else {
          currentChunk = split;
        }
      } else {
        // Add to current chunk
        currentChunk +=
          (currentChunk ? separatorToAdd : "") + split;
      }
    }
    
    // Add remaining chunk
    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }
  }

  /**
   * Split text by character as last resort
   */
  private splitByCharacter(text: string, chunks: string[]): void {
    let start = 0;
    
    while (start < text.length) {
      const end = Math.min(start + this.chunkSize, text.length);
      const chunk = text.slice(start, end);
      
      if (chunk.trim()) {
        chunks.push(chunk.trim());
      }
      
      // Move start position with overlap
      start = end - this.chunkOverlap;
      if (start < 0) start = end;
    }
  }
}

/**
 * Text splitter tool for splitting text into chunks before embedding
 * Uses RecursiveCharacterTextSplitter to intelligently split text
 */
export const textSplitterTool = tool({
  description:
    "Split text into smaller chunks using RecursiveCharacterTextSplitter algorithm. Use this tool before embedding large texts to break them into manageable pieces. The splitter tries to preserve semantic meaning by splitting on paragraphs, sentences, and words in that order.",
  inputSchema: z.object({
    text: z
      .string()
      .describe("The text content to split into chunks"),
    chunkSize: z
      .number()
      .optional()
      .default(1000)
      .describe(
        "Maximum size of each chunk in characters (default: 1000)"
      ),
    chunkOverlap: z
      .number()
      .optional()
      .default(200)
      .describe(
        "Number of characters to overlap between chunks (default: 200)"
      ),
  }),
  execute: async ({ text, chunkSize, chunkOverlap }) => {
    try {
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: chunkSize ?? 1000,
        chunkOverlap: chunkOverlap ?? 200,
      });

      const chunks = splitter.splitText(text);

      if (chunks.length === 0) {
        return {
          success: false,
          error: "No chunks generated from the input text",
          chunks: [],
          count: 0,
        };
      }

      return {
        success: true,
        message: `Successfully split text into ${chunks.length} chunk(s)`,
        chunks: chunks,
        count: chunks.length,
        chunkSize: chunkSize ?? 1000,
        chunkOverlap: chunkOverlap ?? 200,
      };
    } catch (error) {
      return {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to split text. Please try again later.",
        chunks: [],
        count: 0,
      };
    }
  },
});

