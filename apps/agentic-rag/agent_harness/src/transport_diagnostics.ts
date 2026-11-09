export type TransportStage = "fetch" | "response" | "stream";
export type TransportCategory = "dns" | "connect" | "tls" | "timeout" | "abort" | "http" | "unknown";
export interface TransportDiagnostic { stage: TransportStage; category: TransportCategory; error_code: string | null; http_status: number | null; }

// Codes are emitted only on exact matches. Error messages, headers, URLs and
// arbitrary error properties never enter this projection.
const categories: Readonly<Record<string, TransportCategory>> = {
  ENOTFOUND: "dns", EAI_AGAIN: "dns", EAI_FAIL: "dns",
  ECONNREFUSED: "connect", ECONNRESET: "connect", ENETUNREACH: "connect", EHOSTUNREACH: "connect", EPIPE: "connect", UND_ERR_SOCKET: "connect",
  CERT_HAS_EXPIRED: "tls", CERT_NOT_YET_VALID: "tls", DEPTH_ZERO_SELF_SIGNED_CERT: "tls", SELF_SIGNED_CERT_IN_CHAIN: "tls", UNABLE_TO_VERIFY_LEAF_SIGNATURE: "tls", UNABLE_TO_GET_ISSUER_CERT_LOCALLY: "tls", ERR_TLS_CERT_ALTNAME_INVALID: "tls", ERR_SSL_WRONG_VERSION_NUMBER: "tls",
  ETIMEDOUT: "timeout", UND_ERR_CONNECT_TIMEOUT: "timeout", UND_ERR_HEADERS_TIMEOUT: "timeout", UND_ERR_BODY_TIMEOUT: "timeout",
  ABORT_ERR: "abort", UND_ERR_ABORTED: "abort",
};
export function transportDiagnostic(stage: TransportStage, error?: unknown, status?: number, category: TransportCategory = "unknown"): TransportDiagnostic {
  const http_status = Number.isInteger(status) && status! >= 100 && status! <= 599 ? status! : null;
  if (stage === "response" && http_status !== null && http_status >= 300) return { stage, category: "http", error_code: null, http_status };
  if (stage === "fetch") {
    let current: any = error;
    for (let depth = 0; current && typeof current === "object" && depth < 4; depth++, current = current.cause) {
      const code = current.code;
      if (typeof code === "string" && Object.hasOwn(categories, code)) return { stage, category: categories[code]!, error_code: code, http_status };
      if (current.name === "AbortError") category = "abort";
      else if (current.name === "TimeoutError") category = "timeout";
    }
  }
  return { stage, category, error_code: null, http_status };
}
