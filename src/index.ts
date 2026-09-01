export { HarnessError } from './errors.js';
export type { HarnessErrorCode, HarnessErrorOptions } from './errors.js';

export type { JsonObject, JsonValue } from './json.js';
export { isJsonObject } from './json.js';

export type { Durability, SessionStore } from './stores/session-store.js';
export { isDurable } from './stores/session-store.js';
export { MemorySessionStore } from './stores/memory-session-store.js';
export { FileSessionStore } from './stores/file-session-store.js';
export { SessionStoreManager, SLOT_DURABLE, SLOT_EPHEMERAL } from './stores/store-manager.js';
export type { StoreFactory, StoreManagerOptions } from './stores/store-manager.js';

export { Session } from './session.js';
export type { Participant, RunState, SessionOptions } from './session.js';

export { Thread } from './thread.js';
export type { ThreadMessage } from './thread.js';

export { PendingSession, PrismHarness } from './harness.js';
export type { HarnessOptions } from './harness.js';

export { AgentMode, ModeRegistry } from './modes.js';
export type { ModeConfig, ModeRegistryConfig } from './modes.js';

export { MAX_DEPTH, RunBudget, RunContext, RunLedger, Subagent, subagentFromConfig } from './subagents.js';

export { ToolAuthorizer, ToolRegistry, authorizedTool } from './tools.js';
export type {
  CallPolicy,
  HarnessTool,
  OfferPolicy,
  ToolAuthorizerOptions,
  ToolFactory,
  ToolProvider,
} from './tools.js';

export { HarnessEvents } from './events.js';
export type { HarnessEvent, HarnessListener, RunFailed, RunFinished, RunStarted } from './events.js';

export { diagnose } from './doctor.js';
export type { DoctorFinding, DoctorReport, DoctorSubjects } from './doctor.js';

export { AgentRuntime, recordApproval } from './runtime.js';
export type {
  AgentResponse,
  AgentRuntimeOptions,
  LlmClient,
  LlmRequest,
  LlmResponse,
  LlmToolCall,
  PendingApproval,
} from './runtime.js';
