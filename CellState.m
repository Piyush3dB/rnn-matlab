classdef CellState
    properties
        % Recurrent Cell state
        h;
    end
    
    methods
        % Constructor
        function obj = CellState(hidden_size, num_cells)
            obj.h = zeros(hidden_size, num_cells);
        end
    end
end