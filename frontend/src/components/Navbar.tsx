import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Logo } from "./Logo";
import { Github, BookOpen, Settings } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useTheme } from "./theme-provider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

/**
 * Sticky navigation bar with glassmorphism effect
 * Features logo, navigation links, and settings panel
 */
export const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [compactMode, setCompactMode] = useState(false);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };

    // Load settings from localStorage
    setCompactMode(localStorage.getItem('stem-bot-compact') === 'true');

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleCompactModeChange = (checked: boolean) => {
    setCompactMode(checked);
    localStorage.setItem('stem-bot-compact', checked.toString());
  };

  return (
    <motion.nav
      className={`fixed top-0 left-0 right-0 z-30 transition-smooth ${
        isScrolled 
          ? 'glass-medium shadow-glass border-b border-glass-border' 
          : 'bg-transparent'
      }`}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Logo showText={false} />
          
          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-4">
            <Button variant="ghost" size="sm" className="text-secondary hover:text-primary">
              <BookOpen className="h-4 w-4 mr-2" />
              Docs
            </Button>
            <Button variant="ghost" size="sm" className="text-secondary hover:text-primary">
              <Github className="h-4 w-4 mr-2" />
              GitHub
            </Button>
            
            {/* Settings Sheet */}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="sm" className="text-secondary hover:text-primary">
                  <Settings className="h-4 w-4" />
                </Button>
              </SheetTrigger>
              <SheetContent className="glass-strong border-glass-border">
                <SheetHeader>
                  <SheetTitle className="text-primary">Settings</SheetTitle>
                  <SheetDescription className="text-secondary">
                    Customize your STEM Bot experience
                  </SheetDescription>
                </SheetHeader>
                
                <div className="space-y-6 mt-6">
                  {/* Theme Selection */}
                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-primary">Appearance</Label>
                    <Select value={theme} onValueChange={setTheme}>
                      <SelectTrigger className="glass-light border-glass-border">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="light">Light</SelectItem>
                        <SelectItem value="dark">Dark</SelectItem>
                        <SelectItem value="system">Auto</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Compact Mode */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-sm font-medium text-primary">Compact Mode</Label>
                      <p className="text-xs text-secondary">Reduces spacing in the interface</p>
                    </div>
                    <Switch
                      checked={compactMode}
                      onCheckedChange={handleCompactModeChange}
                    />
                  </div>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>

      {/* Gradient border animation */}
      {isScrolled && (
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.3 }}
        />
      )}
    </motion.nav>
  );
};