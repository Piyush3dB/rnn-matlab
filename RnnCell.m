classdef RnnCell
    
    properties
        state;
        params;
        grads;
        loss;
        ps;
        hs;
        ys;
        dhnext;
        dy;
        
    end
    
    methods
        function obj = RnnCell(params)
            obj.params = params;
            obj.state  = CellState(params.hidden_size, 1);
            obj.loss = 0;
        end
        
        function obj = step(obj, inputs)
            
            hprev = obj.state.h;
            
            hidden_size = obj.params.hidden_size;
            seq_length  = obj.params.seq_length;
            vocab_size  = obj.params.vocab_size;
            
            
            obj.loss = 0;
            hs = [hprev, zeros(hidden_size, seq_length)];
            ys = zeros(vocab_size, seq_length);
            ps = zeros(vocab_size, seq_length);
            
            [Wxh, Whh, Why, bh, by] = obj.params2many(obj.params);
            
            %% forward pass
            nFP = seq_length;
            for t = 1:nFP % iterate seq_length
                
                % Input to hidden
                x2h = Wxh * inputs(:, t);
                
                % Hidden to hidden
                h2h = Whh * hs(:, t);
                
                % Update hidden
                hs(:, t+1) = tanh(x2h + h2h + bh);
                
                % Hidden to output
                h2y = Why * hs(:, t+1);
                ys(:, t) = h2y + by;
                
                % Output soft-max and normalise
                ps(:, t) = exp(ys(:, t)) / sum(exp(ys(:, t)));
                
            end
            
            % This gets returned
            obj.state.h = hs(:, end);
            obj.hs = hs;
            obj.ps = ps;
            
        end
        
        function obj = computeLoss(obj, targets)
            
            
            %% Loss function compute
            seq_length  = obj.params.seq_length;
            
            nFP = seq_length;
            for t = 1:nFP % iterate seq_length
                
                % targets(t) contains the index of the target prediction in ps
                tp = obj.ps(targets(t), t);
                
                % cross-entropy loss summation
                obj.loss = obj.loss - log(tp);
                
                obj.dy(t) = tp - 1;
            end
            
        end
        
        function obj = bptt(obj, inputs, targets)
            
            
            hidden_size = obj.params.hidden_size;
            seq_length  = obj.params.seq_length;
            vocab_size  = obj.params.vocab_size;
            
            [Wxh, Whh, Why, bh, by] = obj.params2many(obj.params);
            
            % Struct of grads computed during backward pass
            grads = struct('dWxh', zeros(size(Wxh)), ...
                'dWhh', zeros(size(Whh)), ...
                'dWhy', zeros(size(Why)), ...
                'dbh' , zeros(size(bh )), ...
                'dby' , zeros(size(by )));
            
            obj.dhnext = zeros(hidden_size, 1);
            
            %% BPTT number of forward passes
            hs = obj.hs;
            
            seq_length  = obj.params.seq_length;
            
            nFP = seq_length;
            for t = nFP:-1:1
                
                % Softmax probabilities
                dy = obj.ps(:, t);
                
                % Get index of target prediction
                tpi = targets(t);
                
                % backprop for y
                dy(tpi) = dy(tpi) - 1;
                
                % backprop into h
                dh = Why' * dy + obj.dhnext;
                
                % backprop tanh non-linearity
                dhraw = (1 - (hs(:, t+1) .* hs(:, t+1))) .* dh;
                
                obj.dhnext = Whh' * dhraw;
                
                dWhy = dy * hs(:, t+1)';
                dby  = dy;
                dWhh = dhraw * hs(:, t)';
                dWxh = dhraw * inputs(:, t)';
                
                
                
                grads.dWhy = grads.dWhy + dWhy;
                grads.dby  = grads.dby  + dby;
                grads.dWhh = grads.dWhh + dWhh;
                grads.dbh  = grads.dbh  + dhraw;
                grads.dWxh = grads.dWxh + dWxh;
                
                
                
            end
            
            % clip and prevent exploding gradients
            grads.dWxh = max(min(grads.dWxh, 1), -1);
            grads.dWhh = max(min(grads.dWhh, 1), -1);
            grads.dWhy = max(min(grads.dWhy, 1), -1);
            grads.dbh  = max(min(grads.dbh,  1), -1);
            grads.dby  = max(min(grads.dby,  1), -1);
            
            obj.grads = grads;
            
            
        end
        
        function backprop(obj, dy)
            
            
            % backprop into h
            dh = Why' * dy + obj.dhnext;
            % backprop tanh non-linearity
            dhraw = (1 - (hs(:, t+1) .* hs(:, t+1))) .* dh;
            
            obj.dhnext = Whh' * dhraw;
            
            dWhy = dy * hs(:, t+1)';
            dby  = dy;
            dWhh = dhraw * hs(:, t)';
            dWxh = dhraw * inputs(:, t)';
            
            
            
            obj.grads.dWhy = dWhy;
            obj.grads.dby  = dby;
            obj.grads.dWhh = dWhh;
            obj.grads.dhraw  = dhraw;
            obj.grads.dWxh = dWxh;
            
        end
        
        function ixes = sample(obj, seed, nsample)
            % hprev  \in R (hidden_size, 1)
            % seed   \in {0,1} (vocab_size, 1)  initial seed character (one-hot encoded)
            % nsample -- number of characters to sample
            
            vocab_size = obj.params.vocab_size;
            
            h = obj.state.h;
            
            [Wxh, Whh, Why, bh, by] = obj.params2many(obj.params);
            
            ixes = zeros(1, nsample);
            
            for t = 1:nsample
                
                % update hidden state
                x2h = Wxh * seed;
                h2h = Whh * h;
                h   = tanh(x2h + h2h + bh);
                
                % get prediction scores
                h2y = Why * h;
                y   = h2y + by;
                
                % soft-max probabilities
                pr = exp(y) / sum(exp(y));
                assert(sum(pr) - 1 < 1e-10, 'SoftMax probabilities broken!');
                
                ixes(t) = obj.numpy_random_choice(pr);
                
                seed = zeros(vocab_size, 1);
                seed(ixes(t)) = 1;
            end
        end
        
        function obj = resetState(obj)
            
            obj.state = CellState(obj.params.hidden_size, 1);
            % obj.loss  = 0; TODO?
        end
        
        function obj = ApplyParams(obj, params)
            
            obj.params = params;
            
        end
        
        
    end
    
    methods(Static)
        
        function [Wxh, Whh, Why, bh, by] = params2many(params)
            Wxh = params.Wxh;
            Whh = params.Whh;
            Why = params.Why;
            bh  = params.bh;
            by  = params.by;
        end
        
        function params = many2params(Wxh, Whh, Why, bh, by)
            params.Wxh = Wxh;
            params.Whh = Whh;
            params.Why = Why;
            params.bh  = bh;
            params.by  = by;
        end
        
        
        
        function pick = numpy_random_choice(probabilities)
            %NUMPY_RANDOM_CHOICE
            % Implements Python equivalent of numpy.random.choice()
            %
            %   probabilities: a list of probabilities summing up to 1
            %
            % Example:
            %     probabilities = [0; 0; 0.2; 0.6; 0.2; 0];
            %     for k = 1:1000
            %         X(k) = numpy_random_choice(probabilities);
            %     end
            %     hist(X, 1:6);
            %
            
            cdf = [0; cumsum(probabilities)];
            pick = sum(cdf <= rand);
            
        end
    end
end