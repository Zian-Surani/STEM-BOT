import { motion } from "framer-motion";
import { Logo } from "./Logo";

/**
 * Footer component with branding and privacy information
 * Features glassmorphism design and simple layout
 */
export const Footer = () => {
  return (
    <footer className="mt-20 border-t border-glass-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <motion.div
          className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
        >
          {/* Logo and tagline */}
          <div className="flex items-center space-x-4">
            <Logo showText={false} />
            <div className="text-center md:text-left">
              <p className="text-sm text-secondary">
                Made for students
              </p>
              <p className="text-xs text-muted">
                Instant answers for Science • Math • Social Science
              </p>
            </div>
          </div>

          {/* Privacy and info */}
          <div className="text-center md:text-right">
            <p className="text-xs text-muted">
              Privacy-focused • No data collection • Open source
            </p>
            <p className="text-xs text-muted mt-1">
              © 2024 STEM Bot. Built with care for education.
            </p>
          </div>
        </motion.div>
      </div>
    </footer>
  );
};