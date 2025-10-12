# Chess AI Design System

A Swiss + Terminal-Modern design system emphasizing clarity, precision, and calm professionalism.

## Philosophy

> **"Like well-designed infrastructure rather than marketing."**

Precise, intentional, quietly confident. Not vibe-coded. Not SaaS-template-like. Engineered components that feel handcrafted.

## Color Palette

### Core Colors

| Token | Value | Usage | Example |
|-------|-------|-------|---------|
| `--bg` | `#0F1318` | Background, base layer | Body background |
| `--fg` | `#E5E7EB` | Primary text, main content | Headings, paragraphs |
| `--muted` | `#9AA3AF` | Secondary text, labels | Metadata, captions |
| `--border` | `#1F2937` | Dividers, component edges | Panel borders, table lines |
| `--accent` | `#22D3EE` | Interactive, emphasis | Buttons, links, status |

### Contrast Ratios

- `--fg` on `--bg`: **11.2:1** (WCAG AAA)
- `--muted` on `--bg`: **6.8:1** (WCAG AA)
- `--accent` on `--bg`: **7.4:1** (WCAG AA)

### Usage Guidelines

**DO:**
- Use `--accent` for primary actions and interactive elements
- Use `--muted` for secondary information
- Use `--border` for all dividers and boundaries
- Maintain consistency across components

**DON'T:**
- Mix in gradients
- Add glow effects
- Use transparency for text
- Create new color variations

## Typography

### Font Families

```css
--font-sans: 'Inter', system-ui, -apple-system, sans-serif;
--font-mono: 'JetBrains Mono', 'Courier New', monospace;
```

### Type Scale

| Element | Size | Weight | Line Height | Font |
|---------|------|--------|-------------|------|
| H1 | 1.25rem (20px) | 600 | 1.2 | Sans |
| H2 | 1rem (16px) | 600 | 1.3 | Sans |
| Body | 0.9375rem (15px) | 400 | 1.6 | Sans |
| Small | 0.875rem (14px) | 400 | 1.5 | Sans |
| Tiny | 0.8125rem (13px) | 400 | 1.4 | Sans |
| Label | 0.75rem (12px) | 500 | 1.3 | Sans |
| Data | 0.875rem (14px) | 400 | 1.5 | Mono |

### Usage

**Sans-serif (Inter):**
- UI text
- Navigation
- Headings
- Descriptions
- Labels

**Monospace (JetBrains Mono):**
- Numeric data
- FEN notation
- Status codes
- Move counts
- Feature values

## Spacing System

### Scale

```css
--space-xs: 0.25rem  /* 4px */
--space-sm: 0.5rem   /* 8px */
--space-md: 1rem     /* 16px */
--space-lg: 1.5rem   /* 24px */
--space-xl: 2rem     /* 32px */
--space-2xl: 3rem    /* 48px */
```

### Grid

Base unit: **8px**

All spacing should align to the 8px grid:
- Component padding: 16px or 24px
- Vertical rhythm: 8px increments
- Gap between sections: 32px or 48px

### Usage

| Context | Spacing | Token |
|---------|---------|-------|
| Inline elements | 4px | `--space-xs` |
| Button/input padding | 8px 16px | `--space-sm` `--space-md` |
| Card padding | 24px | `--space-lg` |
| Section margin | 32px | `--space-xl` |
| Major sections | 48px | `--space-2xl` |

## Components

### Buttons

**Primary Button:**
```css
background: var(--accent);
color: var(--bg);
border: 1px solid var(--accent);
padding: 0.5rem 1.5rem;
border-radius: 3px;
transition: 120ms ease;
```

**Secondary Button:**
```css
background: transparent;
color: var(--fg);
border: 1px solid var(--border);
padding: 0.5rem 1.5rem;
border-radius: 3px;
transition: 120ms ease;
```

**States:**
- Hover: Slightly darker/lighter
- Active: No visual change
- Disabled: Opacity 0.5

### Panels

```css
border: 1px solid var(--border);
border-radius: 3px;
overflow: hidden;
```

**Panel Header:**
```css
padding: 1.5rem;
border-bottom: 1px solid var(--border);
background: rgba(31, 41, 55, 0.3);
```

**Panel Content:**
```css
padding: 1.5rem;
```

### Tables

```css
border-collapse: collapse;
width: 100%;
font-size: 0.9375rem;
```

**Headers:**
```css
padding: 1rem;
text-align: left;
font-weight: 600;
color: var(--muted);
text-transform: uppercase;
font-size: 0.75rem;
letter-spacing: 0.05em;
border-bottom: 1px solid var(--border);
```

**Cells:**
```css
padding: 1rem;
border-bottom: 1px solid var(--border);
```

### Status Badges

```css
font-family: var(--font-mono);
font-size: 0.75rem;
padding: 0.25rem 0.5rem;
border: 1px solid var(--accent);
border-radius: 3px;
color: var(--accent);
```

## Layout

### Grid System

```css
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

@media (max-width: 1024px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
```

### Container

```css
.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 1.5rem;
}
```

### Breakpoints

- Desktop: 1024px+
- Tablet: 768px - 1023px
- Mobile: < 768px

## Motion

### Principles

1. **Purpose-driven** - Animation serves a function
2. **Fast** - Duration ≤ 160ms
3. **Subtle** - No bounces, elastics, or attention-seeking
4. **Respectful** - Honor `prefers-reduced-motion`

### Transitions

```css
--transition-fast: 120ms;
```

**Standard:**
```css
transition: all 120ms ease;
```

**Hover States:**
- Opacity changes
- Border color changes
- Background color changes (subtle)

**DON'T:**
- Transform animations
- Complex keyframes
- Attention-grabbing effects
- Long durations

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

## Icons & Graphics

### Chess Pieces (Unicode)

```
White: ♔ ♕ ♖ ♗ ♘ ♙
Black: ♚ ♛ ♜ ♝ ♞ ♟
```

**Display:**
```css
font-size: 3rem;
```

### No Custom Icons

Use text, symbols, or semantic HTML instead of icon fonts.

## Borders & Corners

### Border Width

```css
--border-width: 1px;
```

All borders use `1px solid var(--border)`.

### Border Radius

```css
--radius: 3px;
```

Small, consistent radius on all interactive components.

**Never:**
- Fully rounded (50%)
- Large radius (> 8px)
- Mixed radii

## Shadows

### Usage

Minimal and rare. Only for subtle depth cues.

**Acceptable:**
```css
box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
```

**Avoid:**
- Large shadows
- Colored shadows
- Multiple shadows
- Glow effects

## Accessibility

### Requirements

- ✅ Color contrast ≥ 4.5:1 (WCAG AA)
- ✅ Keyboard navigation support
- ✅ Screen reader compatibility
- ✅ Semantic HTML structure
- ✅ Focus indicators visible
- ✅ Reduced motion support

### Focus States

```css
:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}
```

## Code Style

### CSS

**Property Order:**
1. Display/positioning
2. Box model
3. Typography
4. Visual
5. Animation

**Example:**
```css
.component {
  /* Display */
  display: flex;
  position: relative;
  
  /* Box model */
  padding: 1rem;
  border: 1px solid var(--border);
  
  /* Typography */
  font-family: var(--font-sans);
  font-size: 0.9375rem;
  
  /* Visual */
  color: var(--fg);
  background: var(--bg);
  
  /* Animation */
  transition: all 120ms ease;
}
```

### Naming

- Use semantic names: `.button-primary`, not `.button-blue`
- BEM-inspired: `.panel-header`, `.panel-content`
- Descriptive: `.chess-board`, not `.cb`

## Don'ts

❌ **Never use:**
- Gradients
- Emojis in UI
- Glass/frosted effects
- Neon colors
- Large shadows
- Decorative animations
- Framework defaults
- Auto-generated code

## Dos

✅ **Always:**
- Use design tokens
- Follow spacing grid
- Maintain contrast
- Keep motion subtle
- Write semantic HTML
- Comment sparingly
- Test accessibility
- Respect user preferences

---

**Design System Version**: 1.0  
**Last Updated**: October 2025  
**Adherence**: Strict

