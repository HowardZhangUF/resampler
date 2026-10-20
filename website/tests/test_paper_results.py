"""Check figures against source CSVs, not a second copy of plotted numbers."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import unittest

SITE=Path(__file__).resolve().parents[1]
ROOT=SITE.parent
sys.path.insert(0,str(SITE))
from summary_io import read_summary


class PaperResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.figures=json.loads((SITE/'dist/data/paper-figures.json').read_text())
        cls.export=json.loads((SITE/'dist/data/results.json').read_text())

    def test_export_preserves_source_values_including_tiny_errors(self):
        for name,rows in self.export['tables'].items():
            source=ROOT/self.export['provenance'][name]['file']
            reference=read_summary(source)
            self.assertEqual(len(rows),len(reference),name)
            for actual,expected in zip(rows,reference):
                for key,value in expected.items():
                    if key=='error':continue  # Native errors are intentionally sanitized.
                    self.assertEqual(actual[key],value,(name,key,actual.get('method')))
        eigen=next(r for r in self.export['tables']['static_summary'] if r['provider']=='custom' and r['method']=='eigen' and r['n']==128 and r['dim']==2 and r['cloud']=='mixture')
        self.assertGreater(eigen['covariance_error_mean'],0)
        self.assertLess(eigen['covariance_error_mean'],1e-14)

    def test_best_means_and_all_methods_match_original_summaries(self):
        for figure in self.figures:
            originals=[]
            for source in figure['sources']:
                file=ROOT/source['path']
                self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),source['sha256'])
                originals.extend(read_summary(file))
            chosen=[r for r in originals if all(r.get(k)==v for k,v in figure['case'].items())]
            available=[r for r in chosen if isinstance(r.get(figure['metric']),(int,float)) and math.isfinite(r[figure['metric']])]
            expected=(max if figure.get('direction')=='higher' else min)(r[figure['metric']] for r in available)
            self.assertEqual(figure['best_value'],expected,figure['id'])
            self.assertEqual(figure['total_methods'],len(chosen),figure['id'])
            self.assertEqual(figure['valid_methods'],len(available),figure['id'])
            self.assertTrue(any(r['provider']=='custom' and r['method']=='eigen' for r in figure['shown']))
            with (SITE/'dist'/figure['csv']).open(newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),len(chosen))

    def test_eight_single_metric_figures_have_shareable_assets(self):
        self.assertEqual(len(self.figures),8)
        for number,figure in enumerate(self.figures,1):
            self.assertEqual(figure['number'],number)
            self.assertIn(figure['metric'],figure['rows'][0])
            self.assertIn('<svg',(SITE/'dist'/figure['image']).read_text())
            self.assertIn('<svg',(SITE/'dist'/figure['mobile_image']).read_text())
            self.assertTrue((SITE/'dist'/figure['pdf']).read_bytes().startswith(b'%PDF'))
            self.assertTrue((SITE/'dist'/figure['png']).read_bytes().startswith(b'\x89PNG'))
            self.assertLessEqual(len(figure['shown']),6)
            self.assertTrue(figure['meaning'] and figure['finding'] and figure['limit'])

    def test_native_scores_are_separate_and_have_no_invented_intervals(self):
        score=next(f for f in self.figures if f['id']=='score')
        self.assertEqual(len(score['sources']),2)
        self.assertNotIn('ci_key',score)
        self.assertEqual(score['best_names'],['Eigen (local)'])
        self.assertEqual(score['case'],{'n':128})

    def test_unsuccessful_ungm_attempts_are_retained(self):
        ungm=next(f for f in self.figures if f['id']=='ungm')
        failures=[r for r in ungm['rows'] if r.get('failure_rate',0)>0]
        self.assertTrue(failures)
        for row in failures:
            self.assertLess(row['successful_trials'],row['trials'])
            self.assertEqual(row['provider'],'pfjax')


if __name__=='__main__':unittest.main()
