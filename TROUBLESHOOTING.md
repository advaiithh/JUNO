# Troubleshooting Guide

## Current Issue: UI Not Responding

### Step 1: Check Browser URL
**IMPORTANT:** Use `localhost` instead of `0.0.0.0`:
- ✅ Correct: `http://localhost:8000/ui/index.html`
- ❌ Wrong: `http://0.0.0.0:8000/ui/index.html`

### Step 2: Test Microphone
1. Open browser console (F12 → Console tab)
2. Click the microphone button
3. Watch for these messages:
   - "Requesting microphone access..."
   - "Microphone access granted"
   - "Recording started"
   - "Stopping recording..."
   - "Sending audio to server..."
   - "Response status: 200"

### Step 3: Check Server Logs
In the terminal where `python server.py` is running, you should see:
```
[INFO] Received audio file, starting transcription...
[INFO] Saved audio to /tmp/..., transcribing...
[INFO] Transcript: [your speech]
[INFO] Processing with LLM...
[INFO] LLM response: [response]
```

### Step 4: Common Issues

#### Issue: "NotAllowedError" or "Permission denied"
**Solution:** 
- Click the lock icon in browser address bar
- Allow microphone permissions
- Refresh the page

#### Issue: "Failed to fetch" or Network error
**Solutions:**
1. Make sure server is running: `python server.py`
2. Use `localhost` not `0.0.0.0`
3. Check firewall isn't blocking port 8000
4. Try: `curl http://localhost:8000/status`

#### Issue: "STT not available"
**Solution:**
```bash
pip install faster-whisper
```

#### Issue: "Cannot connect to Ollama"
**Solution:**
```bash
# Start Ollama in another terminal
ollama serve

# Check if model is installed
ollama list

# If not, install it
ollama pull llama3.1:8b
```

#### Issue: No response after clicking mic
**Check:**
1. Browser console for errors (F12)
2. Server terminal for error messages
3. Microphone is working (test in other apps)
4. Network tab in browser DevTools shows the request

### Step 5: Manual Test
Test the server directly:
```bash
# Test status endpoint
curl http://localhost:8000/status

# Should return JSON with server status
```

### Step 6: Enable Detailed Logging
The UI now has detailed console logging. Check browser console for:
- Microphone access
- Recording status
- Server requests
- Response data
- Any errors

## Quick Fix Checklist

- [ ] Server is running (`python server.py`)
- [ ] Using `http://localhost:8000/ui/index.html` (not 0.0.0.0)
- [ ] Microphone permission granted in browser
- [ ] Browser console open (F12) to see errors
- [ ] Ollama is running (`ollama serve`)
- [ ] STT is installed (`pip install faster-whisper`)

## Still Not Working?

1. **Check browser console** - Look for red error messages
2. **Check server terminal** - Look for error messages
3. **Try a different browser** - Sometimes browser extensions interfere
4. **Check microphone** - Test in another app to confirm it works
5. **Restart server** - Stop (Ctrl+C) and restart `python server.py`
