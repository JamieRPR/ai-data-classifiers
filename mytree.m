classdef mytree
    methods(Static)
        
        function m = fit(train_examples, train_labels)
            
            % Defining an empty node structure which will be used later on.
        
            % The (unique) number of the node within the overall tree structure
			emptyNode.number = [];
            % Any training examples and associated labels the node holds
            % (Examples + Labels)
            emptyNode.examples = [];
            emptyNode.labels = [];
            % A prediction based on any class labels the node holds
            emptyNode.prediction = [];
            % A numeric measure of the impurity of any class labels held by
            % a node (used in deciding whether to split it)
            emptyNode.impurityMeasure = [];
            % If the decision is taken to split a node then it will store 
            % two child nodes and divide its training data between them
            emptyNode.children = {};
            % The particular feature (both the number of its column and the
            % name) and the particular value of that feature, which define 
            % the split (splitFeature/splitFeatureName/splitValue)
            emptyNode.splitFeature = [];
            emptyNode.splitFeatureName = [];
            emptyNode.splitValue = [];

            m.emptyNode = emptyNode;
            
            % Defining and populating the root node of the decision tree
            r = emptyNode;
            r.number = 1;
            r.labels = train_labels;
            r.examples = train_examples;
            r.prediction = mode(r.labels);
            
            % the minimum number of examples nodes must contain before 
            % we'll consider splitting them
            m.min_parent_size = 10;
            m.unique_classes = unique(r.labels);
            m.feature_names = train_examples.Properties.VariableNames;
			m.nodes = 1;
            m.N = size(train_examples,1);

            % Generate the decision tree. Trys to see if node can be split,
            % which of course it can. It will do this recursively until
            % the entire tree has been created
            m.tree = mytree.trySplit(m, r);

        end
        
        function node = trySplit(m, node)

            % Determine whether the node can be split
            if size(node.examples, 1) < m.min_parent_size
				return
            end

            % We want nodes that purely, or predominantly contain a single
            % class. GDI (Gini's Diversity Index) is used as the measure of
            % purity, with 0 denoting a pure node, and 1 denoting an impure
            % node.
            node.impurityMeasure = mytree.weightedImpurity(m, node.labels);

            % Considers splitting on every feature
            for i=1:size(node.examples,2)

				fprintf('evaluating possible splits on feature %d/%d\n', i, size(node.examples,2));
                % The split splits the feature data into two tables, one
                % for the values beneath the given value, and one for the
                % values above the given value. Visualise like a Y/N
                % question on one of the nodes
                
				[ps,n] = sortrows(node.examples,i);
                ls = node.labels(n);
                biggest_reduction(i) = -Inf;
                biggest_reduction_index(i) = -1;
                biggest_reduction_value(i) = NaN;
                
                
                % Considers splitting on every indavidual value of the
                % feature
                for j=1:(size(ps,1)-1)
                    if ps{j,i} == ps{j+1,i}
                        continue;
                    end
                    
                    % The classifier will only split if it reduces the
                    % level of impurity within the classifier. This is done
                    % by calculating the GDI score
                    this_reduction = node.impurityMeasure - (mytree.weightedImpurity(m, ls(1:j)) + mytree.weightedImpurity(m, ls((j+1):end)));
                    
                    if this_reduction > biggest_reduction(i)
                        biggest_reduction(i) = this_reduction;
                        biggest_reduction_index(i) = j;
                    end
                end
				
            end

            % Matching the best decisions for the trees node, so that the
            % tree can be made with the smallest amount of impurity.
            [winning_reduction,winning_feature] = max(biggest_reduction);
            winning_index = biggest_reduction_index(winning_feature);

            % If the result is positive then the split produces a reduction
            % in impurity
            if winning_reduction <= 0
                return
            else
                % Node is split, and the process of dewtermining whether
                % the data should be split again is undertaken upon two
                % childeren nodes of the created node.
                
                [ps,n] = sortrows(node.examples,winning_feature);
                ls = node.labels(n);

                node.splitFeature = winning_feature;
                node.splitFeatureName = m.feature_names{winning_feature};
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;

                node.examples = [];
                node.labels = []; 
                node.prediction = [];

                node.children{1} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{1}.number = m.nodes;
                node.children{1}.examples = ps(1:winning_index,:); 
                node.children{1}.labels = ls(1:winning_index);
                node.children{1}.prediction = mode(node.children{1}.labels);
                
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1; 
                node.children{2}.number = m.nodes;
                node.children{2}.examples = ps((winning_index+1):end,:); 
                node.children{2}.labels = ls((winning_index+1):end);
                node.children{2}.prediction = mode(node.children{2}.labels);
                
                % This ius the recursive part, determining whether the
                % child nodes can spawn more child nodes.
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});
            end

        end
        
        function e = weightedImpurity(m, labels)

            % Used to rescale (normalise) each GDI score. Based upon the
            % probability of you reaching the said node.
            %
            % If and only if the weighted impurity score for the parent is 
            % higher than the combined weighted impurity scores for the 
            % children should we make the split.
            % 
            % The probability of descending into any node is given by the 
            % number of examples in that node divided by the total number 
            % of examples in the tree
            
            weight = length(labels) / m.N;

            summ = 0;
            obsInThisNode = length(labels);
            for i=1:length(m.unique_classes)
                
				pc = length(labels(labels==m.unique_classes(i))) / obsInThisNode;
                summ = summ + (pc*pc);
            
			end
            g = 1 - summ;
            
            e = weight * g;

        end

        
        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            for i=1:size(test_examples,1)
                
				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                this_test_example = test_examples{i,:};
                this_prediction = mytree.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
            
			end
        end

        % descend the trained decision tree, applying the splitting rules
        % defined in each node by consulting its splitFeature and
        % splitValue fields and comparing against the value of the
        % corresponding feature in the current test example, until,
        % eventually, it reaches a leaf node. Once there, it returns the
        % node's class prediction.
        
        function prediction = predict_one(m, this_test_example)
            
			node = mytree.descend_tree(m.tree, this_test_example);
            % The predicion extract by decending the decision tree.
            prediction = node.prediction;
        
        end
        
        % descending the tree and returning the child node that is
        % eventually reached.
        
        function node = descend_tree(node, this_test_example)
            
			if isempty(node.children)
                return;
            else
                if this_test_example(node.splitFeature) < node.splitValue
                    node = mytree.descend_tree(node.children{1}, this_test_example);
                else
                    node = mytree.descend_tree(node.children{2}, this_test_example);
                end
            end
        
		end
        
        % describe a tree:
        function describeNode(node)
            
			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                mytree.describeNode(node.children{1});
                mytree.describeNode(node.children{2});        
            end
        
		end
		
    end
end