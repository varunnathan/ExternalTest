import os
import sys
import json
import gzip
import operator

review_file = sys.argv[1]
output_path = sys.argv[2]
min_count = int(sys.argv[3])
output_path += 'min_count' + str(min_count) + '/'
if not os.path.exists(output_path):
	os.makedirs(output_path)

#read all words, users, products
word_count_map = {}
user_set = set()
product_set = set()
with gzip.open(review_file, 'r') as g:
	for l in g:
		l = json.loads(l)
		user = l.get('reviewerID', '')
		product = l.get('asin', '')
		review_text = l.get('reviewText', '')
		summary = l.get('summary', '')
		user_set.add(user)
		product_set.add(product)
		for term in review_text.strip().split(' '):
			if term not in word_count_map:
				word_count_map[term] = 0
			word_count_map[term] += 1
		for term in summary.strip().split(' '):
			if term not in word_count_map:
				word_count_map[term] = 0
			word_count_map[term] += 1

#filter vocabulary by min_count
delete_key = set()
for key in word_count_map:
	if word_count_map[key] < min_count:
		delete_key.add(key)
#output word, user, product indexes
word_list = list(set(word_count_map.keys()) - delete_key)
with gzip.open(output_path + 'vocab.txt.gz','wt') as fout:
	for word in word_list:
		fout.write(word + '\n')
user_list = list(user_set)
with gzip.open(output_path + 'users.txt.gz','wt') as fout:
	for user in user_list:
		fout.write(user + '\n')
product_list = list(product_set)
with gzip.open(output_path + 'product.txt.gz','wt') as fout:
	for product in product_list:
		fout.write(product + '\n')

#read and output indexed reviews
def index_set(s):
	i = 0
	s_map = {}
	for key in s:
		s_map[key] = str(i)
		i += 1
	return s_map
word_map = index_set(word_list)
user_map = index_set(user_list)
product_map = index_set(product_list)
user_review_seq = {} # recording the sequence of user reviews in time
item_review_seq = {} # recording the sequence of item reviews in time
count_valid_review = 0
with gzip.open(output_path + 'review_text.txt.gz', 'wt') as fout_text, gzip.open(output_path + 'review_u_p.txt.gz', 'wt') as fout_u_p:
	with gzip.open(output_path + 'review_id.txt.gz', 'wt') as fout_id, gzip.open(output_path + 'review_rating.txt.gz', 'wt') as fout_rating:
		with gzip.open(review_file, 'r') as g:
			index = 0
			for l in g:
				l = json.loads(l)
				user = l.get('reviewerID', '')
				product = l.get('asin', '')
				review_text = l.get('reviewText', '')
				summary = l.get('summary', '')
				rating = l.get('overall', None)
				time = l.get('unixReviewTime', None)
				count_words = 0
				for term in summary.strip().split(' '):
					if term in word_map:
						fout_text.write(word_map[term] + ' ')
						count_words += 1
				for term in review_text.strip().split(' '):
					if term in word_map:
						fout_text.write(word_map[term] + ' ')
						count_words += 1

				if count_words > 0:
					if user not in user_review_seq:
						user_review_seq[user] = []
					user_review_seq[user].append((count_valid_review, time))
					if product not in item_review_seq:
						item_review_seq[product] = []
					item_review_seq[product].append((count_valid_review, time))
					fout_text.write('\n')
					fout_u_p.write(user_map[user] + ' ' + product_map[product] + '\n')
					fout_id.write('line_' + str(index) + '\n')
					fout_rating.write(str(rating))						
					count_valid_review += 1
				index += 1
				
# Sort each user's reviews according to time and output to files
review_loc_time_list = [[] for _ in range(count_valid_review)]
with gzip.open(output_path + 'u_r_seq.txt.gz', 'wt') as fout:
	for user in user_list:
		review_time_list = user_review_seq[user]
		user_review_seq[user] = sorted(review_time_list, key=operator.itemgetter(1))
		fout.write(' '.join([str(x[0]) for x in user_review_seq[user]]) + '\n')
		for i in range(len(user_review_seq[user])):
			review_id = user_review_seq[user][i][0]
			time = user_review_seq[user][i][1]
			review_loc_time_list[review_id] = [i]

# Sort each item's reviews according to time and output to files
with gzip.open(output_path + 'p_r_seq.txt.gz', 'wt') as fout:
	for product in product_list:
		review_time_list = item_review_seq[product]
		item_review_seq[product] = sorted(review_time_list, key=operator.itemgetter(1))
		fout.write(' '.join([str(x[0]) for x in item_review_seq[product]]) + '\n')
		for i in range(len(item_review_seq[product])):
			review_id = item_review_seq[product][i][0]
			time = item_review_seq[product][i][1]
			review_loc_time_list[review_id].append(i)
			review_loc_time_list[review_id].append(time)

# Output the location (sorted by time) of each review in the corresponding user review list for quick indexing.
with gzip.open(output_path + 'review_uloc_ploc_and_time.txt.gz', 'wt') as fout:
	for t_l in review_loc_time_list:
		fout.write(' '.join([str(x) for x in t_l]) + '\n')
