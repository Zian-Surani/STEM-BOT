# STEM Bot - AI-Powered Study Assistant

A modern, production-ready website for STEM Bot - an AI-powered assistant that provides instant answers for Science, Math, and Social Science questions.

## ✨ Features

- **🔬 Multi-Subject Support**: Science, Mathematics, and Social Sciences
- **📱 Responsive Design**: iOS-17 inspired glassmorphism UI
- **🤖 Standalone Chat**: Beautiful chat interface ready for backend integration
- **📄 OCR Support**: Upload PDFs and images for instant analysis
- **📊 Export Options**: Download chat history in Markdown or JSON
- **🎨 Beautiful UI**: Premium design with smooth animations
- **🌙 Dark/Light Mode**: Auto-switching theme support
- **⚙️ Customizable**: Adjustable settings and compact mode

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd stem-bot

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Start development server
npm run dev
```

The application will be available at `http://localhost:5173`

## ⚙️ Backend Integration

The chat interface is ready for backend integration. The current implementation includes:

- **Message Handling**: Complete message state management
- **API Ready**: Easy integration points for your backend
- **Export Functions**: Download chat history as Markdown or JSON
- **Settings**: Subject, mode, temperature, and top-k parameter controls

### Integration Points

Replace the mock response in `src/components/ChatInterface.tsx`:

```typescript
// Replace this mock response with your API call
setTimeout(() => {
  const botMessage: Message = {
    id: (Date.now() + 1).toString(),
    text: `Your API response here`,
    isBot: true,
    timestamp: new Date(),
  };
  setMessages(prev => [...prev, botMessage]);
  setIsLoading(false);
}, 1000);
```

## 🛠️ Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS + shadcn/ui components
- **Animations**: Framer Motion
- **State**: React hooks + localStorage
- **Backend Integration**: iframe embedding with query parameters

## 📱 Mobile Support

- Responsive design optimized for all screen sizes
- Touch-friendly controls and gestures
- Adaptive chat dock (full-width on mobile, docked on desktop)
- Mobile-first navigation and settings

## 🎨 Design System

The app uses a comprehensive design system with:

- **Glassmorphism**: iOS-17 inspired glass effects
- **Color Tokens**: HSL-based semantic color system
- **Typography**: System fonts with proper scaling
- **Animations**: Smooth transitions and micro-interactions
- **Components**: Reusable shadcn/ui components with custom variants

## 🚀 Deployment

### Vercel (Recommended)

```bash
# Deploy to Vercel
npm run build
vercel --prod
```

### Netlify

```bash
# Build for production
npm run build

# Deploy dist/ folder to Netlify
```

Set any environment variables your backend will need in your deployment platform.

## 🧪 Development

### Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # shadcn/ui base components
│   ├── Logo.tsx        # STEM Bot logo with animations
│   ├── PrimaryCTA.tsx  # Main action buttons
│   ├── SubjectCard.tsx # Subject selection cards
│   ├── FeatureCard.tsx # Feature showcase cards
│   ├── ChatDock.tsx    # Chat interface modal
│   ├── ControlsBar.tsx # Chat controls
│   ├── Navbar.tsx      # Navigation with settings
│   └── Footer.tsx      # Footer component
├── pages/
│   ├── Index.tsx       # Landing page
│   └── NotFound.tsx    # 404 page
├── hooks/              # Custom React hooks
├── lib/                # Utilities and helpers
└── index.css          # Global styles and design tokens
```

### Design Tokens

All colors, spacing, and effects are defined in `src/index.css`:

```css
:root {
  /* Brand gradients */
  --gradient-brand: linear-gradient(135deg, 59 130 246, 147 51 234);
  
  /* Glass morphism */
  --glass-light: 255 255 255 / 0.1;
  --glass-medium: 255 255 255 / 0.2;
  
  /* Text colors */
  --text-primary: 15 23 42;
  --text-secondary: 71 85 105;
}
```

### Adding New Components

1. Create component in `src/components/`
2. Use design tokens from the CSS system
3. Add proper TypeScript interfaces
4. Include JSDoc comments
5. Export from component file

### Custom Hooks

- `useTheme`: Theme management (light/dark/system)
- `useIsMobile`: Mobile breakpoint detection
- `useToast`: Toast notifications

## 🔧 Customization

### Theming

Modify `src/index.css` to customize:

- Color palettes
- Glass effects
- Gradients
- Typography
- Animations

### Component Variants

Add new button variants in `src/components/ui/button.tsx`:

```typescript
variant: {
  // existing variants...
  "your-variant": "your-tailwind-classes",
}
```

### Settings

The settings panel supports:

- **Theme switching**: Light, dark, or system auto
- **Compact mode**: Reduced spacing and smaller chat dock
- **Streamlit URL**: Runtime configuration of backend URL

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📞 Support

For questions and support:

- 📧 Email: support@stembot.app
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Docs: [Documentation](https://docs.stembot.app)

---

Built with ❤️ for students and educators worldwide.
