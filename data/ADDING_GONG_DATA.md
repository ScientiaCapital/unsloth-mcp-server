# Adding Gong Transcripts to Training Data

## Quick Start (30 seconds)

### Option 1: Paste Method
```bash
cd unsloth-mcp-server/data
python add_gong_transcript.py --paste
```
Then paste your transcript and press Enter twice.

### Option 2: File Method
1. Save transcript to `transcript.txt`
2. Run:
```bash
python add_gong_transcript.py --input transcript.txt
```

## Transcript Format

Copy from Gong and format like this:

```
# Discovery Call - ABC Electric (Won deal)

USER: We're looking at different project management tools. ServiceTitan doesn't really work for our solar installs.

AGENT: I hear that a lot. ServiceTitan is built for pure service companies. What's the biggest gap you're seeing with solar specifically?

USER: After we install a system, we have no way to track its performance. Customers call us when something's wrong, but we'd rather know before they do.

AGENT: That's exactly what Coperniq solves. Let me show you our systems dashboard - it shows real-time performance for every asset you've installed. Green means good, yellow means check it, red means act now.

USER: That would save us so many angry customer calls.

AGENT: Exactly. And it's not just monitoring - it's the complete asset lifecycle. All service history, O&M billing, everything tied to that equipment for 25 years. In ServiceTitan, when a job closes, that record goes cold. In Coperniq, the ASSET lives forever.
```

### Role Keywords (all work):
- **Customer side**: `USER:`, `PROSPECT:`, `CUSTOMER:`, `CLIENT:`
- **Your side**: `AGENT:`, `REP:`, `SDR:`, `SALES:`

## Best Calls to Add

Prioritize transcripts where you:
1. ✅ Handled a tough objection well
2. ✅ Positioned against a competitor effectively
3. ✅ Asked great discovery questions
4. ✅ Closed or advanced the deal

## Batch Processing

Add multiple transcripts from a folder:
```bash
for file in transcripts/*.txt; do
  python add_gong_transcript.py --input "$file"
done
```

## Checking Your Data

View total examples:
```bash
wc -l sft_training_data.jsonl
# Output: 83 sft_training_data.jsonl (number of examples)
```

Preview random example:
```bash
shuf -n 1 sft_training_data.jsonl | python -m json.tool
```

## Category Tags

Use categories to organize:
```bash
python add_gong_transcript.py --input call.txt --category discovery
python add_gong_transcript.py --input call.txt --category objection_handling
python add_gong_transcript.py --input call.txt --category closing
python add_gong_transcript.py --input call.txt --category competitor_servicetitan
```

## After Adding Data

Re-run training with updated data:
```bash
# Copy to examples folder
cp sft_training_data.jsonl ../examples/

# Run training on Colab
python train_coperniq_sft.py
```

## Example: Full Workflow

```bash
# 1. Start the paste tool
python add_gong_transcript.py --paste

# 2. Paste this:
# Best Discovery Call - GreenBox Solar

USER: We've been using spreadsheets and email. It works but it's a mess when we need to find information about old installs.

AGENT: That's a common pain point. When a customer calls about a system you installed 2 years ago, how long does it take to find their information?

USER: Sometimes 20 minutes. We have to dig through old emails and shared drives.

AGENT: What if that took 10 seconds instead? In Coperniq, you search the customer name and see everything - the original install, all service history, current system performance, O&M contract status. It's all connected to the ASSET, not buried in project files.

USER: That would be huge. We've lost customers because we couldn't find their info fast enough.

AGENT: Exactly. And the real magic is proactive service - we show you which systems are underperforming BEFORE customers call angry. That's the asset-centric difference.

# 3. Press Enter twice

# 4. Verify it was added
wc -l sft_training_data.jsonl
# Output: 85 (was 83, now +2 examples from this call)
```

## Tips

1. **Quality over quantity**: 10 great transcripts > 100 mediocre ones
2. **Focus on YOUR responses**: The model learns from what you said
3. **Include the "aha moment"**: Where the prospect understood the value
4. **Add context**: The `# title` helps you remember which call it was
5. **Review before training**: Check `sft_training_data.jsonl` for quality
