export interface LocalLLMSupport {
    supported: boolean;
    reason?: string;
    totalMemMB: number;
}

export interface LoadModelOptions {
    /** Absolute filesystem path to the .gguf file (file:// prefix is stripped). */
    modelPath: string;
    /** Context window in tokens. Default 4096. */
    contextSize?: number;
    /** CPU threads. Default: min(4, cores - 2). */
    threads?: number;
}

export interface GenerateOptions {
    systemPrompt?: string;
    prompt: string;
    /** Default 320. */
    maxTokens?: number;
    /** Default 0.7; 0 or less switches to greedy decoding. */
    temperature?: number;
    /** Default 0.9. */
    topP?: number;
    stopSequences?: string[];
}

export interface GenerateResult {
    text: string;
    tokensGenerated: number;
    durationMs: number;
}

export interface InspectionDefect {
    description: string;
    severity?: string;
    quantity?: number;
    comment?: string;
}

export interface InspectionFailedCheckpoint {
    title: string;
    status?: string;
    comment?: string;
}

export interface InspectionComment {
    author?: string;
    text: string;
}

export interface InspectionSummaryData {
    title?: string;
    productName?: string;
    supplierName?: string;
    conclusion?: string;
    defects?: InspectionDefect[];
    failedCheckpoints?: InspectionFailedCheckpoint[];
    comments?: InspectionComment[];
}

export interface SummarizeOptions {
    language?: "en" | "zh";
    maxTokens?: number;
    temperature?: number;
    stopSequences?: string[];
    limits?: {
        maxDefects?: number;
        maxCheckpoints?: number;
        maxComments?: number;
        itemChars?: number;
        totalChars?: number;
    };
}

export interface LocalLLMPlugin {
    isSupported(): Promise<LocalLLMSupport>;
    loadModel(options: LoadModelOptions): Promise<void>;
    unloadModel(): Promise<void>;
    generate(options: GenerateOptions, onToken?: (text: string) => void): Promise<GenerateResult>;
    cancel(): Promise<void>;
    summarizeInspection(
        data: InspectionSummaryData,
        options?: SummarizeOptions,
        onToken?: (text: string) => void,
    ): Promise<GenerateResult>;
}

declare global {
    interface CordovaPlugins {
        localLLM: LocalLLMPlugin;
    }
}
