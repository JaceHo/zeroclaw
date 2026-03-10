import {
  createContext,
  useContext,
  useState,
  useRef,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';
import React from 'react';
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

  const wsRef = useRef<WebSocketClient | null>(null);

  useEffect(() => {
    const ws = new WebSocketClient();

    ws.onOpen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onClose = () => {
      setConnected(false);
    };

    ws.onError = () => {
      setError('Connection error. Attempting to reconnect...');
    };

    ws.onMessage = (msg: WsMessage) => {
      switch (msg.type) {
        case 'chunk':
          setTyping(true);
          setStreamingContent((prev) => prev + (msg.content ?? ''));
          break;

        case 'message':
        case 'done': {
          setStreamingContent((current) => {
            const content = msg.full_response ?? msg.content ?? current;
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
            return '';
          });
          setTyping(false);
          break;
        }

        case 'tool_call':
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: 'agent',
              content: `[Tool Call] ${msg.name ?? 'unknown'}(${JSON.stringify(msg.args ?? {})})`,
              timestamp: new Date(),
            },
          ]);
          break;

        case 'tool_result':
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: 'agent',
              content: `[Tool Result] ${msg.output ?? ''}`,
              timestamp: new Date(),
            },
          ]);
          break;

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
    } catch {
      setError('Failed to send message. Please try again.');
    }
  }, []);

  const value: ChatState = {
    messages,
    streamingContent,
    typing,
    connected,
    error,
    sendMessage,
  };

  return React.createElement(ChatContext.Provider, { value }, children);
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
