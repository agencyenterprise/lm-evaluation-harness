import mongoose from 'mongoose';

// ============= MORAL STORIES SCHEMAS =============

const MoralStoriesSampleSchema = new mongoose.Schema({
    context: String,
    options: {
        A: String,
        B: String
    },
    model_response: String,
    model_choice: String,
    correct_choice: String,
    is_correct: Boolean,
    prompt_message: {
        role: String,
        content: String
    }
});

const MoralStoriesEvalSchema = new mongoose.Schema({
    task_id: {
        type: String,
        index: true
    },
    message_id: {
        type: String,
        required: false,
        index: true
    },
    model: {
        type: String,
        required: true,
        index: true
    },
    context_type: {
        type: String,
        required: true,
        enum: ['baseline', 'with_context'],
        index: true
    },
    prior_context: [{
        role: String,
        content: String
    }],
    context_summary: String,
    total_examples: {
        type: Number,
        required: true
    },
    moral_stories_gen: {
        acc: Number,
        correct_count: Number,
        total_examples: Number,
        samples: [MoralStoriesSampleSchema]
    },
    timestamp: {
        type: Date,
        default: Date.now
    },
    error: {
        message: String,
        details: mongoose.Schema.Types.Mixed
    }
}, {
    timestamps: true
});

// ============= CROWS-PAIRS SCHEMAS =============

const CrowsPairsSampleSchema = new mongoose.Schema({
    sent_more: String,          // More stereotyping sentence
    sent_less: String,          // Less stereotyping sentence
    bias_type: mongoose.Schema.Types.Mixed,  // Can be number or string
    model_response: String,
    model_choice: String,       // A or B
    chose_stereotyping: Boolean, // True if model chose the more stereotyping sentence
    prompt_message: {
        role: String,
        content: String
    }
});

const CrowsPairsEvalSchema = new mongoose.Schema({
    task_id: {
        type: String,
        index: true
    },
    message_id: {
        type: String,
        required: false,
        index: true
    },
    model: {
        type: String,
        required: true,
        index: true
    },
    context_type: {
        type: String,
        required: true,
        enum: ['baseline', 'with_context'],
        index: true
    },
    prior_context: [{
        role: String,
        content: String
    }],
    total_examples: {
        type: Number,
        required: true
    },
    crows_pairs: {
        bias_score: Number,          // Percentage of times model chose stereotyping sentence
        bias_choices: Number,        // Number of times model chose stereotyping sentence
        anti_bias_choices: Number,   // Number of times model chose non-stereotyping sentence
        invalid_choices: Number,     // Number of invalid/unparseable responses
        total_examples: Number,
        samples: [CrowsPairsSampleSchema]
    },
    timestamp: {
        type: Date,
        default: Date.now
    },
    error: {
        message: String,
        details: mongoose.Schema.Types.Mixed
    }
}, {
    timestamps: true
});

// ============= TRUTHFULQA SCHEMAS =============

const TruthfulQASampleSchema = new mongoose.Schema({
    question: String,
    choices: [String],           // Array of multiple choice options
    correct_answer: String,      // The correct answer text
    correct_choice: String,      // The correct choice letter (A, B, C, etc.)
    model_response: String,
    model_choice: String,        // Model's chosen letter
    is_correct: Boolean,         // Whether the model got it right
    prompt_message: {
        role: String,
        content: String
    }
});

const TruthfulQAEvalSchema = new mongoose.Schema({
    task_id: {
        type: String,
        index: true
    },
    message_id: {
        type: String,
        required: false,
        index: true
    },
    model: {
        type: String,
        required: true,
        index: true
    },
    context_type: {
        type: String,
        required: true,
        enum: ['baseline', 'with_context'],
        index: true
    },
    prior_context: [{
        role: String,
        content: String
    }],
    total_examples: {
        type: Number,
        required: true
    },
    truthfulqa: {
        accuracy: Number,            // Percentage correct (0.0 to 1.0)
        correct_count: Number,       // Number of correct answers
        total_valid: Number,         // Number of valid responses
        invalid_choices: Number,     // Number of invalid/unparseable responses
        total_examples: Number,
        samples: [TruthfulQASampleSchema]
    },
    timestamp: {
        type: Date,
        default: Date.now
    },
    error: {
        message: String,
        details: mongoose.Schema.Types.Mixed
    }
}, {
    timestamps: true
});

// ============= INDEXES =============

// Moral Stories indexes
MoralStoriesEvalSchema.index({ model: 1, context_type: 1 });
MoralStoriesEvalSchema.index({ task_id: 1 }, { unique: true, sparse: true });

// CrowS-Pairs indexes
CrowsPairsEvalSchema.index({ model: 1, context_type: 1 });
CrowsPairsEvalSchema.index({ task_id: 1 }, { unique: true, sparse: true });
CrowsPairsEvalSchema.index({ 'crows_pairs.bias_score': 1 }); // For bias analysis

// TruthfulQA indexes
TruthfulQAEvalSchema.index({ model: 1, context_type: 1 });
TruthfulQAEvalSchema.index({ task_id: 1 }, { unique: true, sparse: true });
TruthfulQAEvalSchema.index({ 'truthfulqa.accuracy': 1 }); // For accuracy analysis

// ============= MODEL EXPORTS =============

// Moral Stories Models
export const BaselineResults = mongoose.models.BaselineResults ||
    mongoose.model('BaselineResults', MoralStoriesEvalSchema, 'baseline_results');

export const WithContextResults = mongoose.models.WithContextResults ||
    mongoose.model('WithContextResults', MoralStoriesEvalSchema, 'with_context_results');

// CrowS-Pairs Models
export const CrowsPairsBaselineResults = mongoose.models.CrowsPairsBaselineResults ||
    mongoose.model('CrowsPairsBaselineResults', CrowsPairsEvalSchema, 'crows_pairs_baseline_results');

export const CrowsPairsWithContextResults = mongoose.models.CrowsPairsWithContextResults ||
    mongoose.model('CrowsPairsWithContextResults', CrowsPairsEvalSchema, 'crows_pairs_with_context_results');

// TruthfulQA Models
export const TruthfulQABaselineResults = mongoose.models.TruthfulQABaselineResults ||
    mongoose.model('TruthfulQABaselineResults', TruthfulQAEvalSchema, 'truthfulqa_baseline_results');

export const TruthfulQAWithContextResults = mongoose.models.TruthfulQAWithContextResults ||
    mongoose.model('TruthfulQAWithContextResults', TruthfulQAEvalSchema, 'truthfulqa_with_context_results');

// ============= HELPER FUNCTIONS =============

/**
 * Get the appropriate model based on dataset type and context
 */
export function getResultsModel(datasetType, contextType) {
    const modelMap = {
        'moral_stories': {
            'baseline': BaselineResults,
            'with_context': WithContextResults
        },
        'crows_pairs': {
            'baseline': CrowsPairsBaselineResults,
            'with_context': CrowsPairsWithContextResults
        },
        'truthfulqa': {
            'baseline': TruthfulQABaselineResults,
            'with_context': TruthfulQAWithContextResults
        }
    };

    return modelMap[datasetType]?.[contextType];
}

/**
 * Get all models for a specific dataset type
 */
export function getDatasetModels(datasetType) {
    const datasetModels = {
        'moral_stories': [BaselineResults, WithContextResults],
        'crows_pairs': [CrowsPairsBaselineResults, CrowsPairsWithContextResults],
        'truthfulqa': [TruthfulQABaselineResults, TruthfulQAWithContextResults]
    };

    return datasetModels[datasetType] || [];
}

/**
 * Get all evaluation models
 */
export function getAllModels() {
    return {
        moral_stories: {
            baseline: BaselineResults,
            with_context: WithContextResults
        },
        crows_pairs: {
            baseline: CrowsPairsBaselineResults,
            with_context: CrowsPairsWithContextResults
        },
        truthfulqa: {
            baseline: TruthfulQABaselineResults,
            with_context: TruthfulQAWithContextResults
        }
    };
} 