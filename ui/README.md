# Ruckus UI

A React TypeScript web interface for the Ruckus ML benchmarking system.

## Features

- **Agent Management**: Register, unregister, and monitor ML agents
- **Real-time Monitoring**: Live status updates and polling
- **Interactive Table**: Sortable and reorderable columns
- **Agent Details**: View system information, models, and frameworks
- **Responsive Design**: Clean, professional interface

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
```bash
cp .env.example .env
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Configuration

### Environment Variables

The application can be configured using environment variables. Create a `.env` file in the root directory:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VITE_RUCKUS_SERVER_URL` | URL of the Ruckus server for API calls | `http://localhost:8000` | No |

#### Example `.env` file:
```bash
# Ruckus server configuration
VITE_RUCKUS_SERVER_URL=http://localhost:8000

# For production deployment
# VITE_RUCKUS_SERVER_URL=https://ruckus.example.com

# For different port
# VITE_RUCKUS_SERVER_URL=http://localhost:3000
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint (if configured)

## Architecture

### Project Structure

```
src/
├── components/          # React components
│   ├── AgentsTab.tsx   # Main agents management component
│   └── *.css           # Component styles
├── hooks/              # Custom React hooks
│   ├── useTableSort.ts # Table sorting logic
│   └── useColumnReorder.ts # Column reordering logic
├── services/           # API services
│   └── api.ts          # Ruckus server API client
├── types/              # TypeScript type definitions
│   └── api.ts          # API models and interfaces
├── utils/              # Utility functions
│   └── format.ts       # Data formatting helpers
└── App.tsx             # Main application component
```

### API Integration

The UI communicates with the Ruckus server via REST API:

- `GET /api/v1/agents/` - List registered agents
- `GET /api/v1/agents/status` - Get agent status information
- `POST /api/v1/agents/register` - Register a new agent
- `POST /api/v1/agents/unregister` - Unregister an agent

## Development

### Adding New Features

1. **New Tab**: Add to the `tabs` array in `App.tsx` and create corresponding component
2. **API Endpoints**: Extend the `RuckusApiClient` class in `src/services/api.ts`
3. **Type Definitions**: Add interfaces to `src/types/api.ts`
4. **Styling**: Use CSS modules or add to component-specific CSS files

### Environment Variables in Development

During development, Vite automatically loads environment variables from:
1. `.env.local` (ignored by git)
2. `.env`
3. `.env.development` (for development-specific config)

All environment variables must be prefixed with `VITE_` to be accessible in the client code.

## Deployment

### Production Build

```bash
npm run build
```

The build output will be in the `dist/` directory.

### Environment Variables in Production

Set environment variables before building:

```bash
# Docker example
ENV VITE_RUCKUS_SERVER_URL=https://ruckus-api.production.com

# Or using build-time variables
VITE_RUCKUS_SERVER_URL=https://api.example.com npm run build
```

### Static File Serving

The built application is a static Single Page Application (SPA) that can be served by any web server:

- Nginx
- Apache
- Express.js with `express.static`
- CDN (CloudFront, Netlify, Vercel)

Make sure to configure the server to serve `index.html` for all routes (SPA routing).

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure the Ruckus server allows requests from the UI domain
2. **API Connection Failed**: Check that `VITE_RUCKUS_SERVER_URL` is set correctly
3. **Build Errors**: Verify all TypeScript types are correct and imports use proper syntax

### Debugging

- Check the browser console for client-side errors
- Use browser DevTools Network tab to inspect API requests
- Verify environment variables are loaded: `console.log(import.meta.env)`