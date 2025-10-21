import { motion } from "framer-motion";
import { Bot, User } from "lucide-react";

interface ChatMessageProps {
  message: string;
  isBot: boolean;
  timestamp: Date;
}

/**
 * Individual chat message component
 * Displays messages with proper styling for bot vs user
 */
export const ChatMessage = ({ message, isBot, timestamp }: ChatMessageProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 ${isBot ? 'justify-start' : 'justify-end'}`}
    >
      {isBot && (
        <div className="w-8 h-8 rounded-full gradient-brand flex items-center justify-center shadow-md flex-shrink-0">
          <Bot className="h-4 w-4 text-white" />
        </div>
      )}
      
      <div className={`max-w-[85%] ${isBot ? 'order-2' : 'order-1'}`}>
        <div
          className={`rounded-2xl px-4 py-3 ${
            isBot
              ? 'glass-medium border border-glass-border text-primary'
              : 'gradient-brand text-white shadow-glass'
          }`}
        >
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message}</p>
        </div>
        
        <div className={`flex items-center gap-2 mt-1 px-2 ${isBot ? 'justify-start' : 'justify-end'}`}>
          {!isBot && (
            <div className="w-6 h-6 rounded-full bg-muted flex items-center justify-center">
              <User className="h-3 w-3 text-muted-foreground" />
            </div>
          )}
          <span className="text-xs text-muted">
            {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>
      </div>
    </motion.div>
  );
};