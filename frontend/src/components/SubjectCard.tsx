import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SubjectCardProps {
  title: string;
  subtitle: string;
  icon: LucideIcon;
  onClick?: () => void;
  className?: string;
  delay?: number;
}

/**
 * Subject card component for Sci-Chat, Cal-Chat, and Socio-Bot
 * Features glassmorphism design with hover animations
 */
export const SubjectCard = ({ 
  title, 
  subtitle, 
  icon: Icon, 
  onClick,
  className = "",
  delay = 0
}: SubjectCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      whileHover={{ y: -8, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`group cursor-pointer ${className}`}
    >
      <div className="glass-medium rounded-2xl p-6 h-full shadow-glass hover:shadow-lg transition-smooth border border-glass-border">
        {/* Icon container */}
        <div className="mb-4">
          <div className="w-12 h-12 rounded-xl gradient-brand flex items-center justify-center shadow-md group-hover:shadow-lg transition-smooth">
            <Icon className="h-6 w-6 text-white" />
          </div>
        </div>
        
        {/* Content */}
        <div className="space-y-3">
          <h3 className="text-xl font-semibold text-primary group-hover:text-accent transition-smooth">
            {title}
          </h3>
          <p className="text-secondary text-sm leading-relaxed">
            {subtitle}
          </p>
          
          {/* Learn more button */}
          <Button
            variant="glass-secondary"
            size="sm"
            onClick={onClick}
            className="mt-4 opacity-0 group-hover:opacity-100 transition-smooth"
          >
            Learn more
          </Button>
        </div>
        
        {/* Subtle tilt effect on hover */}
        <motion.div
          className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary/5 to-accent/5 opacity-0 group-hover:opacity-100 transition-smooth"
          style={{ transform: "perspective(1000px) rotateX(2deg) rotateY(-2deg)" }}
        />
      </div>
    </motion.div>
  );
};