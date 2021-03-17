% Naive Bayes works out the probability of each class having a
% feature of value X. Though it assumes that these values are
% independant of each other. Then it adds up all of the
% probabilities for each feature and the data is classified by
% looking at which class it has the highest probability of being.

classdef mynb
    methods(Static)
        
        % Responsible for all of the steps taken when training the
        % classifier. Assumes features are normally distributed. The X axis
        % represents the features value, with the Y axis representing the
        % likelihood.
        
        
        function m = fit(train_examples, train_labels)
            
            % Finds each possible unique classes label
            m.unique_classes = unique(train_labels);
            % Find the total number of different classes
            m.n_classes = length(m.unique_classes);
            
            m.means = {};
            m.stds = {};
            
            for i = 1:m.n_classes
                % Selects a specific class
				this_class = m.unique_classes(i);
                % Pulls all examples of the class from the training data
                examples_from_this_class = train_examples{train_labels==this_class,:};
                % Here the mean and standard deviations are calculated,
                % which will later be used to calculate the normal
                % distrbutions
                m.means{end+1} = mean(examples_from_this_class);
                m.stds{end+1} = std(examples_from_this_class);
			end
            
            m.priors = [];
            
            for i = 1:m.n_classes
                % Selects a specific class
				this_class = m.unique_classes(i);
                % Pulls all examples of the class from the training data
                examples_from_this_class = train_examples{train_labels==this_class,:};
                % Sorts the means and standard deviations into a
                % convienient array
                m.priors(end+1) = size(examples_from_this_class,1) / size(train_labels,1);
			end

        end

        function predictions = predict(m, test_examples)

            predictions = categorical;

            for i=1:size(test_examples,1)
				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                % Isolating a single piece of data in preperation for
                % testing
                this_test_example = test_examples{i,:};
                % Determines the class of the data
                this_prediction = mynb.predict_one(m, this_test_example);
                % Adds the prediction to an array
                predictions(end+1) = this_prediction;
			end
        end

        % Calculates a likelihood for the current test example given the
        % class.
        
        function prediction = predict_one(m, this_test_example)

            for i=1:m.n_classes
                % Considers the value of every feature in isolation (Hence naive).
                %
                % The assumption says indvdual fetures independent if same
                % class value
                %
                % This is the class-conditional independence assumption of
                % Naive Bayes in action.
                %
                % We treat the observed value of each feature as an
                % independent event, then look at how likely it was to 
                % occur given the probability density function we estimated
                % from the distribution of this feature in examples 
                % belonging to this class, and multiply each of the answers
                % we get (one per feature) together to produce a single 
                % estimate of the likelihood of the example given the class.
                % Probability Density = test examples * class * feature
                % values
				this_likelihood = mynb.calculate_likelihood(m, this_test_example, i);
                this_prior = mynb.get_prior(m, i);
                
                % Calculates the probability of the data belonging to each
                % of the available classes
                posterior_(i) = this_likelihood * this_prior;
            
            end

            % The class with the highest probability is assigned to the
            % data and thus the data is classified.
            [winning_value_, winning_index] = max(posterior_);
            prediction = m.unique_classes(winning_index);

        end
        
        function likelihood = calculate_likelihood(m, this_test_example, class)
            
			likelihood = 1;
            
			for i=1:length(this_test_example)
                % Using the probability density (mathematical model of
                % normal distribution) to calculate the likelihood
                % (probability) that the data belongs to a class
                likelihood = likelihood * mynb.calculate_probability_density(this_test_example(i), m.means{class}(i), m.stds{class}(i));
            end
        end
        
        % Used for describing the normal disbuition graph in a
        % mathematical way
        
        function pd = calculate_probability_density(x, mu, sigma)
        
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
        
        end
        
        function prior = get_prior(m, class)
            
			prior = m.priors(class);
		end
 
    end
end