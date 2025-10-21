import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
  delay?: number;
  className?: string;
}

/**
 * Feature card component for showcasing STEM Bot capabilities
 * Used in the features section with animations and glassmorphism
 */
export const FeatureCard = ({ 
  title, 
  description, 
  icon: Icon, 
  delay = 0,
  className = ""
}: FeatureCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ scale: 1.02 }}
      className={`group ${className}`}
    >
      <div className="glass-light rounded-2xl p-6 h-full shadow-md hover:shadow-glass transition-medium border border-glass-border">
        {/* Animated icon */}
        <motion.div
          className="mb-4"
          whileHover={{ scale: 1.1, rotate: 5 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-smooth">
            <Icon className="h-5 w-5 text-primary" />
          </div>
        </motion.div>
        
        {/* Content */}
        <div className="space-y-2">
          <h3 className="text-lg font-semibold text-primary">
            {title}
          </h3>
          <p className="text-secondary text-sm leading-relaxed">
            {description}
          </p>
        </div>
        
        {/* Hover accent line */}
        <motion.div
          className="mt-4 h-0.5 bg-gradient-to-r from-primary to-accent rounded-full"
          initial={{ width: 0 }}
          whileHover={{ width: "100%" }}
          transition={{ duration: 0.3 }}
        />
      </div>
    </motion.div>
  );
};