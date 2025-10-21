/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_STEMBOT_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}