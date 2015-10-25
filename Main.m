% Convert from Andrej's Python NumPY gist to Matlab
% https://gist.github.com/karpathy/d4dee566867f8291f086
%
dbstop if error
%Set seed for repeatability
randn('seed',0)

input_fname = 'ex1.txt';
fid = fopen(input_fname, 'r');

% Load contents of text file
data = textscan(fid, '%c', 'whitespace', '', 'delimiter', '\n');
fclose(fid);

% integer data (ASCII values, easier to get one-hot vectors with arrayfun :))
idata = double(data{1})';

% integer unique characters (ASCII).
% Used to get dictionary for 1-of-k encoding
ichars = unique(idata);
fprintf('%d unique characters [%s] in data.\n', length(ichars), string(ichars));

%% Setup parameters
%%% Data parameters

% Size of unique number of characters
vocab_size = length(ichars);

% Total input data size
data_size  = length(idata);

%%% Hyperparameters
hidden_size   = 100;      % size of hidden layer of neurons
seq_length    = 25;        % number of RNN unroll steps aka batch size
lr            = 1e-1;   % learning rate



%% Input and Target data preparation
% ichars is dictionary
% char_to_ix (1 of k encoding)
fn = @(x) ichars == x;
inputs = arrayfun(fn, idata, 'UniformOutput', false);
encInputs = double(cat(1, inputs{:})');

% Target data preparation
% Find the dictionary index of input characters.
fn = @(x) find(ichars == x);
encTargets = arrayfun(fn, idata);

%% Visualisation
figure(1);
clf;
h_iters = [];
h_sloss = [];
h_loss  = [];

figure(2);
clf;

%% Start learning
epochs = 0; % initialise num epochs
n = 0;      % iteration counter
p = 1;      % data pointer

%% Initialise loss and other
loss = -log(1/vocab_size) * seq_length;
smooth_loss = loss;
b = 0.999; a = [1 -1+b];
%hist = loss;
%[smooth_loss, hist] = filter(b, a, loss, hist);

%% RNN TRAIN

PARAMS = RnnParams(hidden_size, vocab_size, seq_length);
RNN    = RnnCell(PARAMS);


carryOn = true;
while carryOn
    
    %%% reset after one pass over all data OR at the first iteration
    if p+seq_length >= data_size || n == 0
        epochs = epochs + 1;
        %fprintf('>>>> Starting %d epoch... \n', epochs);
        RNN    = RNN.resetState();      % reset RNN memory
        
        p = 1;                              % move data pointer to start
    end
    
    %%% get inputs and targets
    inputs = encInputs(:,p : p+seq_length-1);
    targets = encTargets(:, p+1 : p+seq_length);
    
    %%% sample from model
    if mod(n, 100) == 0
        sample_ix = RNN.sample(inputs(:, 1), 200);
        text = char(ichars(sample_ix));
        fprintf(2, '--- Sampled text @ iter = %5d | loss = %2.2f ---  ', n, smooth_loss);
        fprintf('%s -----------\n', text);
        
        % Plot figure;
        h_iters = [h_iters ; n];
        h_sloss = [h_sloss ; smooth_loss];
        h_loss  = [h_loss  ; loss];
        
        % Loss function
        h = figure(1);
        plot(h_iters, h_sloss, '*-'); hold on;
        plot(h_iters, h_loss , '.-r');
        title('loss function')
        drawnow;
        set(0, 'CurrentFigure', h)
        
        % Histogram of coefficients
        
        figure(2);
        %{
        subplot(4,1,1);
        plot(h_iters, h_sloss, '*-'); hold on;
        plot(h_iters, h_loss , '.-r');
        title('loss function')
        %}
        
        %{
        subplot(4,1,2);
        rWxh = reshape(PARAMS.Wxh, size(PARAMS.Wxh,1)*size(Wxh,2), 1);
        hist(rWxh, 100, 'r');
        title('hist Wxh');
        xlim([-4 4]);
        
        subplot(4,1,3);
        rWhh = reshape(Whh, size(Whh,1)*size(Whh,2), 1);
        hist(rWhh, 50, 'r');
        title('hist Whh');
        xlim([-4 4]);
        
        subplot(4,1,4);
        rWhy = reshape(Why, size(Why,1)*size(Why,2), 1);
        hist(rWhy, 50, 'FaceColor','r');
        title('hist Why');
        xlim([-4 4]);
        
        %}
        
    end
    
    %%% forward seq_length characters and get gradients
    RNN = RNN.step(inputs);
    RNN = RNN.computeLoss(targets);
    RNN = RNN.bptt(inputs, targets);
    
    smooth_loss = smooth_loss * 0.999 + RNN.loss * 0.001;
    loss = RNN.loss;
    %PARAMS = PARAMS.LossUpdate(RNN.loss);
    
    %%% Use gradients to update parameters with Adagrad
    PARAMS = PARAMS.ParamUpdate(RNN.grads, lr);
    RNN = RNN.ApplyParams(PARAMS);
    
    
    
    %%% Update counters
    p = p + seq_length;
    n = n + 1;
    
    if n == 10000
        carryOn = false;
    end
end



disp('Main.m done')
