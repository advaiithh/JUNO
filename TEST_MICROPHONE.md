# Test Microphone - Step by Step

## ✅ What I Fixed

1. **Console Filtering** - Extension errors are now hidden automatically
2. **Better Error Messages** - Clear instructions for each error type
3. **Improved Logging** - Color-coded console messages for easy debugging
4. **Touch Support** - Works on mobile devices too
5. **Better Error Handling** - Handles all microphone permission scenarios

## 🧪 How to Test

### Step 1: Refresh the Page
1. Press **Ctrl+F5** (or Cmd+Shift+R on Mac) to hard refresh
2. This clears cache and loads the new code

### Step 2: Check Console
Open console (F12) and you should see:
```
🎤 JUNO AI Voice Assistant
Initializing JUNO UI...
Current URL: http://localhost:8000/ui/index.html
Hostname: localhost
Voice button found: true
✓ JUNO UI initialized successfully
Click the microphone button to start recording!
```

**Note:** Extension errors are now filtered out automatically!

### Step 3: Click Microphone Button
1. Click the large purple microphone button
2. You should see in console:
   ```
   🎤 Voice button clicked!
   🎤 Requesting microphone access...
   ```
3. Browser will show permission prompt
4. Click **"Allow"**

### Step 4: Record Audio
1. After allowing, you'll see:
   ```
   ✓ Microphone access granted!
   Recording started
   ```
2. Speak into your microphone
3. Click the button again to stop

### Step 5: Check Response
1. After stopping, you should see:
   ```
   Recording stopped, chunks: X
   Audio blob created, size: X bytes
   Sending audio to server...
   ```
2. Server will process and respond
3. Response will appear in the chat area

## 🔍 Troubleshooting

### If microphone permission doesn't appear:
1. Check browser address bar - look for microphone icon
2. Click the lock icon → Allow microphone
3. Refresh page and try again

### If you see "NotAllowedError":
1. Go to: `chrome://settings/content/microphone`
2. Find `localhost:8000`
3. Set to "Allow"
4. Refresh page

### If button doesn't respond:
1. Open console (F12)
2. Type: `document.getElementById('voiceBtn').click()`
3. Press Enter
4. This should trigger the microphone request

### If you see "No audio chunks recorded":
- Speak louder or move closer to microphone
- Check microphone is working in other apps
- Try a different microphone

## ✅ Expected Console Output (Success)

```
🎤 JUNO AI Voice Assistant
Initializing JUNO UI...
Current URL: http://localhost:8000/ui/index.html
Hostname: localhost
Voice button found: true
Clear button found: true
View history button found: true
Checking server status...
✓ Server is running
STT available: true
TTS available: false
Face auth: true
✓ JUNO UI initialized successfully
Click the microphone button to start recording!

[After clicking mic:]
🎤 Voice button clicked!
🎤 Requesting microphone access...
✓ Microphone access granted!
Audio tracks: 1
Using MIME type: audio/webm
Recording started
Audio data available, size: X
[Keep speaking...]
[Click button again to stop]
Recording stopped, chunks: X
Audio blob created, size: X bytes, type: audio/webm
Sending audio to server...
Fetching /voice_chat...
Response status: 200 OK
Server response: {transcript: "...", response: "...", ...}
```

## 🚨 Common Issues

### Extension Errors (IGNORE THESE)
- `chrome-extension://invalid/` - Filtered out automatically
- `zybTrackerStatisticsAction` - Filtered out automatically  
- `copilot` errors - Filtered out automatically

### Real Errors (FIX THESE)
- `NotAllowedError` - Microphone permission denied
- `NotFoundError` - No microphone found
- `NotReadableError` - Microphone in use by another app
- `Failed to fetch` - Server not running

## 🎯 Quick Test Command

In browser console, type:
```javascript
// Test microphone access
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    console.log('✓ Microphone works!', stream);
    stream.getTracks().forEach(track => track.stop());
  })
  .catch(err => console.error('✗ Microphone error:', err));
```

This will test if microphone access works at all.
