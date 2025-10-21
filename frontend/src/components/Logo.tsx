import { motion } from "framer-motion";

interface LogoProps {
  className?: string;
  showText?: boolean;
}

/**
 * STEM Bot logo with SVG mark and optional text
 * Features gradient colors and subtle animations
 */
export const Logo = ({ className = "", showText = true }: LogoProps) => {
  return (
    <motion.div
      className={`flex items-center gap-3 ${className}`}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Logo Mark - STEM letters with orbit */}
      <div className="relative">
        <svg 
          width="32" 
          height="32" 
          viewBox="0 0 32 32" 
          fill="none" 
          xmlns="http://www.w3.org/2000/svg"
          className="text-primary"
        >
          {/* Background circle */}
          <circle 
            cx="16" 
            cy="16" 
            r="15" 
            fill="url(#logoGradient)" 
            className="opacity-10"
          />
          
          {/* STEM text stylized */}
          <text 
            x="16" 
            y="20" 
            textAnchor="middle" 
            fontSize="10" 
            fontWeight="bold" 
            fill="url(#logoGradient)"
            fontFamily="system-ui"
          >
            STEM
          </text>
          
          {/* Orbit ring */}
          <motion.circle 
            cx="16" 
            cy="16" 
            r="12" 
            stroke="url(#logoGradient)" 
            strokeWidth="1.5" 
            fill="none" 
            strokeDasharray="4 4"
            initial={{ rotate: 0 }}
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            style={{ originX: "16px", originY: "16px" }}
          />
          
          {/* Small orbit dot */}
          <motion.circle 
            cx="28" 
            cy="16" 
            r="2" 
            fill="url(#logoGradient)"
            initial={{ rotate: 0 }}
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            style={{ originX: "16px", originY: "16px" }}
          />
          
          <defs>
            <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style={{ stopColor: "#3B82F6" }} />
              <stop offset="100%" style={{ stopColor: "#9333EA" }} />
            </linearGradient>
          </defs>
        </svg>
      </div>
      
      {showText && (
        <div className="flex flex-col">
          <span className="text-xl font-bold text-primary">STEM Bot</span>
          <span className="text-xs text-muted font-medium leading-tight">
            Instant answers for Science • Math • Social Science
          </span>
        </div>
      )}
    </motion.div>
  );
};