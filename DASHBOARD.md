# Matrix-Themed Dashboard

## Overview

The Sequence Framework Matrix Dashboard is an interactive, visually striking monitoring interface that displays real-time training progress, model performance metrics, and system status with a classic Matrix movie aesthetic.

## Features

### Visual Design
- **Matrix Rain Animation**: Cascading green characters create an authentic Matrix background effect
- **Neon Green Theme**: High-contrast green-on-black color scheme with glow effects
- **Animated Scan Line**: Horizontal scan line effect for that retro terminal feel
- **Glowing Text**: All text elements have subtle glow animations for enhanced visibility

### Dashboard Panels

#### 1. System Status
Displays core system information:
- Framework version
- Python runtime version
- PyTorch backend status
- Repository information
- Status badges for key components (Neural Net, CUDA, A3C, GDELT)

#### 2. Training Progress
Real-time training metrics:
- Current epoch progress (37/50)
- Batch processing status (512/1024)
- Animated progress bars with shimmer effects
- Visual completion percentage

#### 3. Model Performance
Live performance metrics with dynamic updates:
- **Accuracy**: Classification accuracy percentage
- **Loss**: Training loss value
- **Precision**: Model precision metric
- **Recall**: Model recall metric

All metrics update every 2 seconds with realistic variations.

#### 4. Active Components
Checklist showing the status of all framework components:
- ✓ CNN Feature Extraction
- ✓ LSTM Temporal Modeling
- ✓ Multi-Head Attention
- ✓ Intrinsic Time Transform
- ✓ FinBERT Sentiment Analysis
- ✓ A3C Execution Policy
- ○ TimesFM Ensemble (pending)
- ✓ Backtesting Integration

#### 5. Data Pipeline
Statistics about data processing:
- Currency pairs being analyzed
- Total data points processed
- GDELT news events integrated
- Sentiment feature status
- Intrinsic time bars status

#### 6. Reinforcement Learning
A3C training progress:
- Algorithm type (Asynchronous Advantage Actor-Critic)
- Total training steps with progress
- Number of active workers
- Average reward metric
- Animated progress visualization

#### 7. System Log Terminal
Scrolling log display with color-coded messages:
- **Green (SUCCESS)**: Successful operations
- **Cyan (INFO)**: Informational messages
- **Yellow (WARNING)**: Warning messages

The log automatically scrolls and maintains the last 20 entries, with new entries appearing every 3 seconds.

## Usage

### Opening the Dashboard

1. **Direct Browser Access**:
   ```bash
   # Navigate to the repository directory
   cd /path/to/Sequence
   
   # Open dashboard.html in your browser
   open dashboard.html  # macOS
   xdg-open dashboard.html  # Linux
   start dashboard.html  # Windows
   ```

2. **Using a Local Web Server**:
   ```bash
   # Start a simple HTTP server
   python3 -m http.server 8080
   
   # Open in browser
   # Navigate to: http://localhost:8080/dashboard.html
   ```

### Customization

The dashboard uses named constants for easy customization. Edit the JavaScript section at the bottom of `dashboard.html`:

```javascript
// Animation settings
const MATRIX_ANIMATION_INTERVAL = 35; // milliseconds between frames
const MAX_LOG_ENTRIES = 20; // maximum log entries to keep
const EPOCH_PROGRESS_RATE = 0.01; // simulated epoch progress increment
const BATCH_PROGRESS_RATE = 0.5; // simulated batch progress increment
```

### Interactive Features

- **Auto-updating Clock**: System time updates every second
- **Live Metrics**: Performance metrics update every 2 seconds
- **Animated Progress**: Progress bars animate smoothly
- **Scrolling Logs**: New log entries appear every 3 seconds
- **Hover Effects**: Panels glow when hovered
- **Responsive Design**: Adapts to different screen sizes

## Technical Details

### Technologies Used
- **HTML5**: Structure and canvas for Matrix rain
- **CSS3**: Styling, animations, and visual effects
- **JavaScript (Vanilla)**: All animations and interactivity
- **No external dependencies**: Completely self-contained

### Performance Optimizations
- Debounced window resize handler (250ms delay)
- Efficient canvas rendering for Matrix rain
- Limited log entries to prevent memory bloat
- CSS-based animations for smooth performance

### Browser Compatibility
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support
- Opera: Full support

Minimum recommended: Modern browsers with Canvas and ES6 support

## Screenshots

![Matrix Dashboard](https://github.com/user-attachments/assets/9db9b17a-2164-4a73-870c-2da5c25f3070)

## Integration with Sequence Framework

The dashboard currently displays simulated data for demonstration purposes. To integrate with real training data:

1. **Training Script Integration**: Update your training scripts to export metrics to a JSON file
2. **Dashboard Data Loading**: Modify the dashboard to load and display real-time metrics
3. **WebSocket Support**: Implement WebSocket communication for live updates
4. **REST API**: Create an API endpoint that the dashboard can poll for updates

Example JSON structure for metrics:
```json
{
  "epoch": 37,
  "total_epochs": 50,
  "batch": 512,
  "total_batches": 1024,
  "accuracy": 0.947,
  "loss": 0.0342,
  "precision": 0.923,
  "recall": 0.958,
  "rl_steps": 742891,
  "rl_workers": 8,
  "rl_avg_reward": 0.0428
}
```

## Contributing

To enhance the dashboard:

1. Maintain the Matrix theme aesthetic
2. Keep animations smooth and non-distracting
3. Ensure all new features are mobile-responsive
4. Test across multiple browsers
5. Document any new configuration options

## License

This dashboard is part of the Sequence Framework and is licensed under the MIT License.

## Credits

Inspired by the iconic Matrix movie franchise aesthetic. Created for the Sequence deep learning framework for FX market prediction.
