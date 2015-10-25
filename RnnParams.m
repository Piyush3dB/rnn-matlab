classdef RnnParams
    
    properties
        hidden_size;
        vocab_size;
        seq_length;
        
        Wxh;
        Whh;
        Why;
        bh;
        by;
        
        mWxh;
        mWhh;
        mWhy;
        mbh;
        mby;
        
        loss;
        smooth_loss;
        
    end
    
    methods
        function obj = RnnParams(hidden_size, vocab_size, seq_length)
            
            obj.loss = -log(1/vocab_size) * seq_length;
            
            obj.smooth_loss = obj.loss;
            
            obj.hidden_size = hidden_size;
            obj.vocab_size  = vocab_size;
            obj.seq_length  = seq_length;
            
            %%% Initialize RNN model parameters - weights for bits
            obj.Wxh = randn(hidden_size, vocab_size )*0.01;   % weights: input to hidden
            obj.Whh = randn(hidden_size, hidden_size)*0.01;   % weights: hidden to hidden
            obj.Why = randn(vocab_size , hidden_size)*0.01;   % weights: hidden to output
            obj.bh  = zeros(hidden_size, 1);                  % bias: hidden
            obj.by  = zeros(vocab_size , 1);                  % bias: output
            
            %%% Memory variables for adagrad - weights for bits
            obj.mWxh = zeros(size(obj.Wxh));
            obj.mWhh = zeros(size(obj.Whh));
            obj.mWhy = zeros(size(obj.Why));
            obj.mbh  = zeros(size(obj.bh));
            obj.mby  = zeros(size(obj.by));
            
        end
        
        function obj = ParamUpdate(obj, grads, lr)
            %%% Use gradients to update parameters with Adagrad
            [obj.mWxh, obj.Wxh] = obj.adagrad_update(lr, obj.Wxh, grads.dWxh, obj.mWxh);
            [obj.mWhh, obj.Whh] = obj.adagrad_update(lr, obj.Whh, grads.dWhh, obj.mWhh);
            [obj.mWhy, obj.Why] = obj.adagrad_update(lr, obj.Why, grads.dWhy, obj.mWhy);
            [obj.mbh , obj.bh ] = obj.adagrad_update(lr, obj.bh , grads.dbh , obj.mbh);
            [obj.mby , obj.by ] = obj.adagrad_update(lr, obj.by , grads.dby , obj.mby);
        end
        
        function obj = LossUpdate(obj, newLoss)
            b = 0.999; a = [1 -1+b];
            obj.smooth_loss = obj.smooth_loss * 0.999 + obj.loss * 0.001;
            
        end
    end
    
    methods(Static)
        %% Support functions
        function [mem, param] = adagrad_update(lr, param, dparam, mem)
            mem   = mem + dparam .* dparam;
            param = param - lr * dparam ./ sqrt(mem + 1e-8);
        end
    end
end