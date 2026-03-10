import {
  createContext,
  useContext,
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from 'react';
import type { WsMessage } from '@/types/api';
import { WebSocketClient } from '@/lib/ws';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: Date;
}

export interface ChatState {
  messages: ChatMessage[];
  /** The partially streamed content for the current response. */
  streamingContent: string;
  typing: boolean;
  connected: boolean;
  error: string | null;
  sendMessage: (content: string) => void;
}

const ChatContext = createContext<ChatState | null>(null);

// ---------------------------------------------------------------------------
// Provider — lives above the router so state survives navigation
// ---------------------------------------------------------------------------

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingContent, setStreamingContent] = useState('');
  const [typing, setTyping] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ref to access current streaming content without stale closures
  const streamingRef = useRef('');

  const wsRef = useRef<WebSocketClient | null>(null);

  useEffect(() => {
    const ws = new WebSocketClient();

    ws.onOpen = () => {
      setConnected(true);
      setError(null);
      // Reset streaming state on (re)connect to prevent stale partial content
      setTyping(false);
      setStreamingContent('');
      streamingRef.current = '';
    };

    ws.onClose = () => {
      setConnected(false);
      // Clean up streaming state so the UI doesn't show a permanently
      // "typing" indicator with stale partial text.
      setTyping(false);
      setStreamingContent('');
      streamingRef.current = '';
    };

    ws.onError = () => {
      setError('Connection error. Attempting to reconnect...');
    };

    ws.onMessage = (msg: WsMessage) => {
      switch (msg.type) {
        case 'chunk':
          setTyping(true);
          streamingRef.current += msg.content ?? '';
          setStreamingContent(streamingRef.current);
          break;

        case 'done': {
          // Use ref to read current streaming content (avoids nested setState).
          const content = msg.full_response ?? msg.content ?? streamingRef.current;
          if (content) {
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: 'agent',
                content,
                timestamp: new Date(),
              },
            ]);
          }
          setStreamingContent('');
          streamingRef.current = '';
          setTyping(false);
          break;
        }

        // Note: tool_call/tool_result are reserved for future protocol
        // extensions. The server currently only sends chunk/done/error.

        case 'error':
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: 'agent',
              content: `[Error] ${msg.message ?? 'Unknown error'}`,
              timestamp: new Date(),
            },
          ]);
          setTyping(false);
          setStreamingContent('');
          streamingRef.current = '';
          break;
      }
    };

    ws.connect();
    wsRef.current = ws;

    return () => {
      ws.disconnect();
    };
  }, []);

  const sendMessage = useCallback((content: string) => {
    const trimmed = content.trim();
    if (!trimmed || !wsRef.current?.connected) return;

    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
        timestamp: new Date(),
      },
    ]);

    try {
      wsRef.current.sendMessage(trimmed);
      setTyping(true);
      setStreamingContent('');
      streamingRef.current = '';
    } catch {
      setError('Failed to send message. Please try again.');
    }
  }, []);

  // Memoize context value to prevent unnecessary re-renders of all consumers
  // when only unrelated state changes.
  const value: ChatState = useMemo(
    () => ({
      messages,
      streamingContent,
      typing,
      connected,
      error,
      sendMessage,
    }),
    [messages, streamingContent, typing, connected, error, sendMessage],
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useChat(): ChatState {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error('useChat must be used within a <ChatProvider>');
  }
  return ctx;
}
