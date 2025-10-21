import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Download, ChevronDown } from "lucide-react";

interface ControlsBarProps {
  subject: string;
  setSubject: (subject: string) => void;
  mode: string;
  setMode: (mode: string) => void;
  temperature: number;
  setTemperature: (temperature: number) => void;
  topK: number;
  setTopK: (topK: number) => void;
  onDownloadMD: () => void;
  onDownloadJSON: () => void;
}

const subjects = [
  { value: "sci", label: "Sci-Chat" },
  { value: "cal", label: "Cal-Chat" },
  { value: "socio", label: "Socio-Bot" },
];

const modes = [
  { value: "mcq", label: "MCQ" },
  { value: "short", label: "Short" },
  { value: "long", label: "Long" },
];

/**
 * Simplified controls bar for standalone chat interface
 */
export const ControlsBar = ({
  subject,
  setSubject,
  mode,
  setMode,
  temperature,
  setTemperature,
  topK,
  setTopK,
  onDownloadMD,
  onDownloadJSON,
}: ControlsBarProps) => {
  return (
    <motion.div
      className="glass-medium rounded-2xl p-4 border border-glass-border"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex flex-wrap items-center gap-4">
        {/* Subject Selection */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-secondary">Subject:</label>
          <Select value={subject} onValueChange={setSubject}>
            <SelectTrigger className="w-32 glass-light border-glass-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {subjects.map((subj) => (
                <SelectItem key={subj.value} value={subj.value}>
                  {subj.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Mode Selection */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-secondary">Mode:</label>
          <Select value={mode} onValueChange={setMode}>
            <SelectTrigger className="w-28 glass-light border-glass-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {modes.map((m) => (
                <SelectItem key={m.value} value={m.value}>
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Temperature Slider */}
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-secondary">Temperature:</label>
          <div className="flex items-center gap-2">
            <Slider
              value={[temperature]}
              onValueChange={(value) => setTemperature(value[0])}
              max={1}
              min={0}
              step={0.1}
              className="w-20"
            />
            <span className="text-xs text-muted w-8">{temperature.toFixed(1)}</span>
          </div>
        </div>

        {/* Top-K Slider */}
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-secondary">Top-k:</label>
          <div className="flex items-center gap-2">
            <Slider
              value={[topK]}
              onValueChange={(value) => setTopK(value[0])}
              max={10}
              min={1}
              step={1}
              className="w-20"
            />
            <span className="text-xs text-muted w-6">{topK}</span>
          </div>
        </div>

        {/* Download Menu */}
        <div className="flex items-center gap-2 ml-auto">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="glass-secondary" size="sm" className="flex items-center gap-2">
                <Download className="h-3 w-3" />
                Export
                <ChevronDown className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="glass-medium border-glass-border">
              <DropdownMenuItem onClick={onDownloadMD}>
                Download as Markdown
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onDownloadJSON}>
                Download as JSON
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </motion.div>
  );
};