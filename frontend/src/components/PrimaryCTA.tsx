import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";

interface PrimaryCTAProps {
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
  size?: "default" | "lg" | "xl";
  disabled?: boolean;
}

/**
 * Primary CTA button with gradient background and motion effects
 * Used for main action buttons like "Dive into STEM BOT"
 */
export const PrimaryCTA = ({ 
  children, 
  onClick, 
  className = "",
  size = "xl",
  disabled = false
}: PrimaryCTAProps) => {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 400, damping: 17 }}
    >
      <Button
        variant="hero"
        size={size}
        onClick={onClick}
        disabled={disabled}
        className={`group relative overflow-hidden ${className}`}
      >
        {/* Ripple effect background */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent"
          initial={{ x: "-100%" }}
          whileHover={{ x: "100%" }}
          transition={{ duration: 0.6 }}
        />
        
        <span className="relative z-10 flex items-center gap-2">
          {children}
          <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
        </span>
      </Button>
    </motion.div>
  );
};