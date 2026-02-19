# Piper TTS Setup Instructions

Piper TTS is required for voice synthesis but is not included in the repository due to size.

## Download Piper

1. Go to: https://github.com/rhasspy/piper/releases
2. Download `piper_windows_amd64.zip` (latest release)
3. Extract the ZIP file
4. Copy the contents to `JUNO/piper/` folder

Your folder structure should look like:
```
JUNO/
└── piper/
    ├── piper.exe
    ├── *.dll files
    ├── espeak-ng-data/ (folder)
    └── voices/ (create this folder)
```

## Download Voice Model

1. Go to: https://huggingface.co/rhasspy/piper-voices/tree/main
2. Navigate to: `en/en_US/lessac/medium/`
3. Download:
   - `en_US-lessac-medium.onnx`
   - `en_US-lessac-medium.onnx.json`
4. Place both files in `JUNO/piper/voices/`

### Alternative Voices

For different voices, browse: https://rhasspy.github.io/piper-samples/

Popular options:
- `en_US-amy-medium` (Female, clear)
- `en_US-lessac-medium` (Male, professional)
- `en_US-ryan-medium` (Male, neutral)

## Verify Installation

Your `piper/` folder should have:
```
piper/
├── piper.exe ✓
├── piper_phonemize.dll ✓
├── espeak-ng.dll ✓
├── espeak-ng-data/ ✓
└── voices/
    ├── en_US-lessac-medium.onnx ✓
    └── en_US-lessac-medium.onnx.json ✓
```

## Test Voice

Run this to test:
```bash
echo "Hello, this is a test" | piper\piper.exe --model piper\voices\en_US-lessac-medium.onnx --output_file test.wav
```

You should hear audio in `test.wav`!

## Troubleshooting

**"piper.exe not found"**
- Make sure you extracted to the correct folder
- Check the path: `JUNO/piper/piper.exe`

**"Voice model not found"**
- Verify files are in `piper/voices/`
- Check file names match exactly

**"DLL missing" error**
- Download the complete release package
- Don't mix files from different releases

## Done!

Once Piper is set up, JUNO will automatically detect and use it for voice synthesis.
