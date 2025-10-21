import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ControlsBar } from "./ControlsBar";
import { ChatInterface } from "./ChatInterface";
import { X } from "lucide-react";

interface ChatDockProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Standalone chat dock component with modern design
 * Centers the chat in a modal overlay
 */
export const ChatDock = ({ isOpen, onClose }: ChatDockProps) => {
  const [subject, setSubject] = useState("sci");
  const [mode, setMode] = useState("short");
  const [temperature, setTemperature] = useState(0.3);
  const [topK, setTopK] = useState(5);

  const handleDownloadMD = () => {
    // Create a simple transcript for demo
    const transcript = `# STEM Bot Chat Transcript\n\nExported: ${new Date().toISOString()}\nSubject: ${subject}\nMode: ${mode}\nTemperature: ${temperature}\nTop-k: ${topK}\n\n*Chat messages would be exported here*`;
    const blob = new Blob([transcript], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stembot-chat-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadJSON = () => {
    const transcript = {
      timestamp: new Date().toISOString(),
      settings: { subject, mode, temperature, topK },
      messages: "Chat messages would be exported here"
    };
    const blob = new Blob([JSON.stringify(transcript, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stembot-chat-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Chat Modal - Centered */}
          <motion.div
            className="fixed inset-4 md:inset-8 lg:inset-16 z-50 flex items-center justify-center"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            <div className="glass-strong rounded-3xl shadow-glow border border-glass-border h-full w-full max-w-4xl flex flex-col overflow-hidden">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-glass-border">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center shadow-lg">
                    <span className="text-white font-bold text-sm">AI</span>
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-primary">STEM Bot</h2>
                    <p className="text-sm text-secondary">Your AI Study Assistant</p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClose}
                  className="h-10 w-10 hover:bg-glass-light rounded-xl"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>

              {/* Controls */}
              <div className="p-6 border-b border-glass-border">
                <ControlsBar
                  subject={subject}
                  setSubject={setSubject}
                  mode={mode}
                  setMode={setMode}
                  temperature={temperature}
                  setTemperature={setTemperature}
                  topK={topK}
                  setTopK={setTopK}
                  onDownloadMD={handleDownloadMD}
                  onDownloadJSON={handleDownloadJSON}
                />
              </div>

              {/* Chat Interface */}
              <div className="flex-1 min-h-0">
                <ChatInterface
                  subject={subject}
                  mode={mode}
                  temperature={temperature}
                  topK={topK}
                />
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};