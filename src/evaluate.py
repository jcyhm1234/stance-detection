import shlex, subprocess, sys

class Evaluate:
	"""
	helps use given script to compute F-scores from predictions and gold labels

	"""
	def computeFscores(self, testData, predstance):
		f = open('../output/tmp_gold','w')
		f.write('ID	Target	Tweet	Stance\n')
		for ind,tw in enumerate(testData):
			f.write(str(ind)+'\t'+tw[1]+'\t'+tw[0]+'\t'+tw[2]+'\n')
		f.close()

		f = open('../output/tmp_pred','w')
		f.write('ID	Target	Tweet	Stance\n')
		for ind,tw in enumerate(testData):
			f.write(str(ind)+'\t'+tw[1]+'\t'+tw[0]+'\t'+predstance[ind]+'\n')
		f.close()

		return self.computeFscoresFromFiles('../output/tmp_gold','../output/tmp_pred')

	def computeFscoresFromFiles(self, goldlabels, predlabels):
		"""
		takes filepaths as arg
		"""
		args_str = "perl ../scripts/eval/eval.pl "+goldlabels+" "+predlabels
		args = shlex.split(args_str)
		p = subprocess.Popen(args, stdout=subprocess.PIPE)
		output = p.communicate()
		p.stdout.close()  # Allow ps_process to receive a SIGPIPE if grep_process exits.
		ans = {}
		for l in output[0].split('\n'):
			words = l.split()
			if words:
				if words[0] == 'FAVOR' or words[0]=='AGAINST':
					ans[words[0]] = {}
					ans[words[0]][words[1]] = float(words[2])
					ans[words[0]][words[3]] = float(words[4])
					ans[words[0]][words[5]] = float(words[6])
				if words[0]=='Macro':
					ans[words[0]] = float(words[2])
		return ans

if __name__=='__main__':
	e = Eval()
	e.computeFscoresFromFiles('../scripts/eval/gold_toy.txt','../scripts/eval/guess_toy.txt')