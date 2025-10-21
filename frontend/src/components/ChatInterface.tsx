import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatMessage } from "./ChatMessage";
import { Send, RotateCcw } from "lucide-react";

interface Message {
  id: string;
  text: string;
  isBot: boolean;
  timestamp: Date;
}

interface ChatInterfaceProps {
  subject: string;
  mode: string;
  temperature: number;
  topK: number;
}

/**
 * Standalone chat interface component
 * Handles message display and user input
 */
export const ChatInterface = ({ subject, mode, temperature, topK }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: `Hello! I'm your ${subject === 'sci' ? 'Science' : subject === 'cal' ? 'Math' : 'Social Science'} assistant. How can I help you today?`,
      isBot: true,
      timestamp: new Date(),
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      isBot: false,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    // Simulate bot response (replace with actual API call)
    setTimeout(() => {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `I understand you're asking about "${inputMessage}". This is where I would provide a detailed ${mode} response about your ${subject === 'sci' ? 'science' : subject === 'cal' ? 'math' : 'social science'} question. The response would be generated using temperature ${temperature} and top-k ${topK} settings.`,
        isBot: true,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
      inputRef.current?.focus();
    }, 1000 + Math.random() * 2000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleNewChat = () => {
    setMessages([
      {
        id: '1',
        text: `Hello! I'm your ${subject === 'sci' ? 'Science' : subject === 'cal' ? 'Math' : 'Social Science'} assistant. How can I help you today?`,
        isBot: true,
        timestamp: new Date(),
      }
    ]);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            message={message.text}
            isBot={message.isBot}
            timestamp={message.timestamp}
          />
        ))}
        
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-3 justify-start"
          >
            <div className="w-8 h-8 rounded-full gradient-brand flex items-center justify-center shadow-md">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            </div>
            <div className="glass-medium border border-glass-border rounded-2xl px-4 py-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-glass-border p-4">
        <div className="flex gap-3 items-end">
          <div className="flex-1">
            <Input
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything about your subject..."
              className="glass-light border-glass-border resize-none"
              disabled={isLoading}
            />
          </div>
          
          <div className="flex gap-2">
            <Button
              variant="glass-secondary"
              size="icon"
              onClick={handleNewChat}
              className="h-10 w-10"
              title="New Chat"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            
            <Button
              variant="hero"
              size="icon"
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="h-10 w-10"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        <p className="text-xs text-muted mt-2 px-1">
          Press Enter to send • Shift+Enter for new line
        </p>
      </div>
    </div>
  );
};