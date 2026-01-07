#!/bin/bash

# Easy training launcher for all 3 model variants (SEQUENTIAL)
# Usage: bash run_all_variants.sh [quick|full]

MODE=${1:-quick}  # default to "quick" if no argument

# === Configuration ===
if [ "$MODE" == "quick" ]; then
    EPOCHS=5
    LIMIT_TRAIN=500
    LIMIT_VAL=100
    echo "🚀 QUICK MODE: epochs=5, limited dataset (500/100)"
elif [ "$MODE" == "full" ]; then
    EPOCHS=10
    LIMIT_TRAIN=2000
    LIMIT_VAL=""
    echo "🚀 FULL MODE: epochs=10, limited to 2000 training samples"
else
    echo "❌ Invalid mode. Use: quick or full"
    exit 1
fi

# Common args - OPTIMIZED FOR SPEED
COMMON="--epochs $EPOCHS --batch-size 32 --lr 3e-4 --wd 1e-4 --device mps"
[ -n "$LIMIT_TRAIN" ] && COMMON="$COMMON --limit-train $LIMIT_TRAIN"
[ -n "$LIMIT_VAL" ] && COMMON="$COMMON --limit-val $LIMIT_VAL"

# Create logs directory
mkdir -p logs

echo "Starting 3 SEQUENTIAL trainings on Apple Silicon (MPS)..."
echo "Settings: batch_size=32, lr=3e-4, epochs=$EPOCHS, limit_train=$LIMIT_TRAIN"
echo ""

# === Train 1: Full model (SWT + Transformer) ===
echo "════════════════════════════════════════════════════════════"
echo "🚀 [1/3] Training FULL model (SWT + Transformer)..."
echo "════════════════════════════════════════════════════════════"
python3 scripts/model_train.py $COMMON \
    --model-variant full \
    --save-dir checkpoints/full \
    2>&1 | tee logs/full.log

if [ $? -eq 0 ]; then
    echo "✅ Full model training completed!"
else
    echo "❌ Full model training failed!"
    exit 1
fi

echo ""

# === Train 2: No SWT (Transformer only) ===
echo "════════════════════════════════════════════════════════════"
echo "🚀 [2/3] Training NO-SWT model (Transformer only)..."
echo "════════════════════════════════════════════════════════════"
python3 scripts/model_train.py $COMMON \
    --model-variant no-swt \
    --save-dir checkpoints/no_swt \
    2>&1 | tee logs/no_swt.log

if [ $? -eq 0 ]; then
    echo "✅ No-SWT model training completed!"
else
    echo "❌ No-SWT model training failed!"
    exit 1
fi

echo ""

# === Train 3: No Transformer (SWT only) ===
echo "════════════════════════════════════════════════════════════"
echo "🚀 [3/3] Training NO-TRANSFORMER model (SWT only)..."
echo "════════════════════════════════════════════════════════════"
python3 scripts/model_train.py $COMMON \
    --model-variant no-transformer \
    --save-dir checkpoints/no_transformer \
    2>&1 | tee logs/no_transformer.log

if [ $? -eq 0 ]; then
    echo "✅ No-Transformer model training completed!"
else
    echo "❌ No-Transformer model training failed!"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🎉 ALL 3 TRAININGS COMPLETED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results saved in:"
echo "   • checkpoints/full/"
echo "   • checkpoints/no_swt/"
echo "   • checkpoints/no_transformer/"
echo ""
echo "📝 Training logs saved in:"
echo "   • logs/full.log"
echo "   • logs/no_swt.log"
echo "   • logs/no_transformer.log"