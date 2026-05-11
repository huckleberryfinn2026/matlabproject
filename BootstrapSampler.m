classdef BootstrapSampler < handle
    % BootstrapSampler  Handle class that manages bootstrap-sampling state.
    %
    % Inherits from handle (Lecture 8: handle classes), so the internal
    % Counter and RNG state persist across method calls without having
    % to reassign the object. This contrasts with value classes (such as
    % OLS.m) where each method call would return a new copy.

    properties
        N           % sample size
        B           % total number of replications
        Counter     % current replication index
    end

    methods

        function obj = BootstrapSampler(N, B, seed)
            % Constructor: store dimensions and seed the RNG.
            obj.N       = N;
            obj.B       = B;
            obj.Counter = 0;
            rng(seed);
        end

        function idx = nextSample(obj)
            % Draw an N-by-1 vector of bootstrap indices and advance the
            % internal counter. Because BootstrapSampler is a handle
            % class, obj.Counter is updated in place.
            obj.Counter = obj.Counter + 1;
            idx         = randi(obj.N, obj.N, 1);
        end

        function reset(obj, seed)
            % Reset the counter and optionally reseed the RNG.
            obj.Counter = 0;
            if nargin > 1
                rng(seed);
            end
        end

    end
end
