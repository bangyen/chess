# Design System

A modern, minimalist design system for the Chess AI dashboard, focused on clarity, professional aesthetics, and high performance.

## Design Tokens

### Colors

| Token | Value | Description |
| :--- | :--- | :--- |
| `--color-bg` | `#F5F7FA` | Light page background |
| `--color-surface` | `#FFFFFF` | Card and surface background |
| `--color-text-primary` | `#1A202C` | High-contrast primary text |
| `--color-text-secondary` | `#4A5568` | Subtle secondary text |
| `--color-accent` | `#2C5F2D` | Chess board green / Primary action |
| `--color-border` | `#E1E8ED` | Component borders and dividers |

### Typography

- **System Stack**: `Space Grotesk`, `Inter`, system-ui
- **Data/Monospace**: `JetBrains Mono`
- **Headings**: Semibold (600) to Bold (700)
- **Body**: Regular (400) to Medium (500)

### Spacing

Base unit: `4px`
- `xs`: 4px
- `sm`: 8px
- `md`: 16px
- `lg`: 24px
- `xl`: 32px

## Components

### Dashboard Cards
Cards use a subtle border instead of heavy shadows for a clean, professional look.
```css
.chart-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  padding: 1.5rem;
}
```

### Buttons
Primary buttons use the accent green with high-contrast white text.
```css
.btn-primary {
  background: var(--color-accent);
  color: white;
  border-radius: 8px;
  font-weight: 500;
}
```

## Accessibility
- **Contrast**: WCAG AA compliant (4.5:1)
- **Motion**: Minimal transitions (120ms)
- **Structure**: Semantic HTML5 with ARIA labels
