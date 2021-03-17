classdef myknn
    methods(Static)
        
        % Takes roughly 75% of the available data to use as the training
        % data. 
        
        
        function m = fit(train_examples, train_labels, k)
            
            % Standardisation normalises the Euclidean distance so that
            % scaling doesnt affect classification. For example, if you had
            % 2 classes with a single feature, and one classes feature
            % values varied greatly, say between 100 & -100, whereas the
            % other vaired between 0 & 1, the first class would distort the
            % classification. 
            
            % start of standardisation process
            
            % We are using Z-Score standardisation as our means of
            % standardisation
            %
            % Z-Score standardisation subtracts the mean value of the feature (so that the feature
            % distribution is centred on zero) and divides by the standard deviation of the feature. 
            % Because the standard deviation gives a measure of the feature's overall spread (how much
            % individual features tend to differ from their mean), features with a big spread get scaled 
            %  down, and features with a small spread get scaled up.
            
			m.mean = mean(train_examples{:,:});
			m.std = std(train_examples{:,:});
            
            for i=1:size(train_examples,1)
				train_examples{i,:} = train_examples{i,:} - m.mean;
                train_examples{i,:} = train_examples{i,:} ./ m.std;
            end
            % end of standardisation process
            
            m.train_examples = train_examples;
            m.train_labels = train_labels;
            m.k = k;
        
        end

        % After training the model we must test the classifier to determine
        % its level of accuracy.
        
        function predictions = predict(m, test_examples)

            predictions = categorical;

            for i=1:size(test_examples,1)
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1));
                
                this_test_example = test_examples{i,:};
                
                % start of standardisation process
                this_test_example = this_test_example - m.mean;
                this_test_example = this_test_example ./ m.std;
                % end of standardisation process
                
                this_prediction = myknn.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
            
            end
        
        end

        % This finds which other data points, which have been determined
        % through the training period, are the closest to the given data
        % point. When this has been determined whichever class is most
        % prominent in the nearest data found is assigned to the data.
        
        function prediction = predict_one(m, this_test_example)
            
            distances = myknn.calculate_distances(m, this_test_example);
            neighbour_indices = myknn.find_nn_indices(m, distances);
            prediction = myknn.make_prediction(m, neighbour_indices);
        
        end

        % 
        % 
        % 
        % 
        
        function distances = calculate_distances(m, this_test_example)
            
			distances = [];
            
			for i=1:size(m.train_examples,1)
                
				this_training_example = m.train_examples{i,:};
                this_distance = myknn.calculate_distance(this_training_example, this_test_example);
                distances(end+1) = this_distance;
            end
        
        end

        % This is the "engine house" function of the classifier, and finds
        % the Euclidean (straight line) between two data points by using the pythagorean theorem  
        
        function distance = calculate_distance(p, q)
            
			differences = q - p;
            squares = differences .^ 2;
            total = sum(squares);
            distance = sqrt(total);
        
        end

        % This function sorts the array of distance near the data points,
        % and finds the K nearest, which will then be used to classify the
        % data point
        
        function neighbour_indices = find_nn_indices(m, distances)
            
			[sorted, indices] = sort(distances);
            neighbour_indices = indices(1:m.k);
        
		end
        
        function prediction = make_prediction(m, neighbour_indices)

			neighbour_labels = m.train_labels(neighbour_indices);
            % find the most common class label amongst the nearest neighbours and store it as our predicted class for this example:
            prediction = mode(neighbour_labels);
        
		end

    end
end

