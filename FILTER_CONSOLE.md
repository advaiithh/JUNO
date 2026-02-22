# Filter Console Errors - Ignore Extension Errors

## Most Errors Are From Browser Extensions (Safe to Ignore)

The errors you're seeing are from browser extensions, NOT from JUNO:

### ❌ Ignore These (Extension Errors):
- `chrome-extension://invalid/` - Broken extension
- `zybTrackerStatisticsAction` - Some tracking extension
- `copilot.b68e6a51.js` - GitHub Copilot extension
- `contentScript.bundle.js` - Some extension's content script
- `PC plat undefined` - Extension error

### ✅ Look For These (JUNO Messages):
- "Initializing JUNO UI..."
- "Voice button clicked"
- "Requesting microphone access..."
- "Microphone access granted"
- "Recording started"
- "Sending audio to server..."

## How to Filter Console

1. Open Console (F12)
2. Click the filter icon (funnel) or type in the filter box
3. Type: `-extension` to hide extension errors
4. Or type: `index.html` to show only JUNO messages

## Test If Microphone Works

1. **Clear the console** (right-click → Clear console)
2. **Click the microphone button**
3. **Look for these messages:**
   - "Voice button clicked, isRecording: false"
   - "Requesting microphone access..."
   - Browser should show permission prompt
   - "Microphone access granted"
   - "Recording started"

## If You Don't See Microphone Messages

The microphone button might not be working. Check:
1. Is the button clickable? (Does it change when you hover?)
2. Do you see "Voice button clicked" in console?
3. If not, there might be a JavaScript error blocking it

## Quick Test

1. Open console (F12)
2. Type this in console and press Enter:
   ```javascript
   document.getElementById('voiceBtn').click()
   ```
3. This should trigger the microphone request
