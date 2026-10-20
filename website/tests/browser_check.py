"""Portable browser QA with a locally installed Chrome. Preview server must be running."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/website-qa';OUT.mkdir(exist_ok=True)
LESSONS=['start','particles','filter','classical','claims','eigen','limits','gradients','metrics','benchmarks','results','resources']
errors=[];failed=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
    page.on('response',lambda r:failed.append(f'{r.status} {r.url}') if r.status>=400 and '127.0.0.1' in r.url else None)
    def go(lesson):
        page.goto(f'http://127.0.0.1:8765/#/{lesson}')
        page.locator('h1').wait_for()
        page.wait_for_function("document.querySelector('h1')?.textContent !== 'Loading the learning guide…'")
        page.wait_for_timeout(80)
    for lesson in LESSONS:
        go(lesson)
        assert page.locator('.nav-link.active').count()==1
        assert page.locator('.quiz-option').count()==3
        assert page.locator('.katex-error').count()==0,lesson
        # No page-level sideways scrolling, including wide equations/tables.
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),lesson
        checks.append('Route: '+lesson)
    assert page.locator('#repo-list a').count()==8
    assert page.locator('#repo-list a').first.get_attribute('href').startswith('https://github.com/')
    go('particles')
    before=page.locator('.lab-metrics').inner_text()
    page.locator('#weight-0').fill('99')
    assert page.locator('.lab-metrics').inner_text()!=before
    page.locator('[data-term="particle"]').click();assert page.locator('.term-popover').is_visible()
    page.keyboard.press('Escape');assert page.locator('.term-popover').count()==0
    page.locator('[data-answer="1"]').click();assert 'Correct.' in page.locator('.quiz-feedback').inner_text()
    page.locator('#complete-lesson').click();page.reload();assert page.locator('#progress-label').inner_text()=='1 / 12'
    checks+=['Weight sliders, glossary keyboard dismissal, quiz feedback, saved progress']
    go('claims');assert page.locator('.worked-example').count()>=1
    assert '1.25' in page.locator('.lab-metrics').inner_text()
    page.locator('#contract-question').select_option('right');page.locator('#contract-p').fill('0.9')
    assert page.locator('#contract-reading').inner_text()
    checks.append('First-principles two-particle claim lab')
    go('metrics');page.locator('#metric-kind').select_option('collapse')
    assert 'uncertainty is lost' in page.locator('#metric-reading').inner_text()
    page.locator('#metric-kind').select_option('eigen');page.locator('#metric-p').fill('0.9')
    page.locator('#metric-kind').select_option('shift');page.locator('#metric-shift').fill('-1.2')
    assert '1.2' in page.locator('.lab-metrics').inner_text()
    checks.append('Metric playground: shift, collapse, and affine errors')
    go('benchmarks')
    for choice in ['moments','shape','score','learning','recovery','sampling']:
        page.locator('#task-question').select_option(choice)
        assert 'Reference:' in page.locator('#task-answer').inner_text()
        assert page.locator('#task-answer a[target="_blank"]').get_attribute('href').startswith('https://github.com/')
    checks.append('All six claim-to-task guides and source links')
    for lesson in LESSONS[4:]:
        go(lesson);assert page.locator('.worked-example').count()>=1,lesson
        assert 'First principle:' in page.locator('.lesson-section h2').first.inner_text(),lesson
        assert page.locator('.katex-error').count()==0,lesson
    checks.append('Lessons 5–12: first principle and worked numerical examples')
    go('classical');page.locator('#repeat').click();assert '500 draws' in page.locator('#batch-status').inner_text()
    page.locator('#class-order').select_option('sorted');page.locator('#class-method').select_option('residual')
    checks.append('Classical draws and repeated sampling')
    go('eigen');page.locator('[data-stage="white"]').click();assert 'Whitened covariance' in page.locator('#eigen-status').inner_text()
    page.locator('[data-stage="y"]').click();assert 'Final moment check' in page.locator('#eigen-status').inner_text()
    page.locator('#eigen-rho').fill('0.9');checks.append('Eigen stage controls')
    go('limits');before=page.locator('.lab-metrics').inner_text();page.locator('#mix-mass').fill('0.9');assert before!=page.locator('.lab-metrics').inner_text()
    page.locator('#mix-method').select_option('gaussian');page.locator('#mix-draw').click();checks.append('Mixture controls and counterexample')
    go('gradients');page.locator('#gradient-function').select_option('square');assert 'x² is fixed' in page.locator('#gradient-status').inner_text()
    page.locator('#gradient-theta').fill('-0.6');page.locator('#gradient-step').select_option('0.00001');checks.append('Gradient objective and finite-difference controls')
    go('filter');page.locator('#pf-time').fill('1');page.locator('#pf-method').select_option('eigen');page.locator('#pf-n').select_option('512');page.locator('#pf-time').fill('40');page.locator('#pf-new').click();checks.append('Filter count, method, time scrub, and new data')
    go('results');page.locator('.paper-result').last.wait_for()
    assert page.locator('figure.paper-figure').count()==8
    assert page.locator('#advanced-explorer').get_attribute('open') is None
    for figure in page.locator('.paper-result').all():
        assert figure.locator('figure img').count()==1
        figure.locator('figure img').scroll_into_view_if_needed()
        page.wait_for_function('(img)=>img.complete && img.naturalWidth>0',arg=figure.locator('figure img').element_handle())
        assert 'Figure ' in figure.locator('figcaption').inner_text()
        assert 'What the metric means' in figure.locator('.paper-metric').inner_text()
        assert 'OBSERVED MEAN' in figure.locator('.paper-best').inner_text()
        assert 'What this benchmark shows.' in figure.locator('.paper-takeaway').inner_text()
        for link in figure.locator('.paper-downloads a').all():
            assert page.request.get('http://127.0.0.1:8765/'+link.get_attribute('href')).ok
    covariance=page.locator('[data-figure="covariance"]')
    assert 'e-15' in covariance.locator('.paper-eigen').inner_text()
    assert '1.31e-15' in covariance.locator('.paper-eigen').inner_text()
    ungm=page.locator('[data-figure="ungm"]');assert 'PFJAX' in ungm.locator('.paper-failures').inner_text()
    score=page.locator('[data-figure="score"]');assert 'Eigen (local)' in score.locator('.paper-best').inner_text()
    page.locator('[data-figure="shape"] .paper-full-table summary').click()
    assert page.locator('[data-figure="shape"] tbody tr').count()==36
    checks+=['Eight single-metric paper figures with caption, metric meaning, best mean, and takeaway','Figure SVG/PDF/PNG/CSV downloads available','Small nonzero errors and failed attempts retained','All-method figure tables available']
    page.locator('#advanced-explorer>summary').click()
    page.locator('#results-table tr').first.wait_for();assert page.locator('#results-table tr').count()>20
    page.locator('#results-provider').select_option('custom');assert page.locator('#results-table tr').count()==3
    page.locator('#results-metric').select_option('covariance_error_mean');page.locator('#results-scale').select_option('log')
    with page.expect_download() as download:page.locator('#filtered-download').click()
    assert 'custom' in Path(download.value.path()).read_text()
    for tab in ['filter','gradient','learning','native']:
        page.locator(f'[data-tab="{tab}"]').click();assert page.locator('#results-table tr').count()>0,tab
        for option in page.locator('#results-metric option').all():
            page.locator('#results-metric').select_option(option.get_attribute('value'))
            assert page.locator('#results-status').inner_text()
    checks.append('All result datasets and metrics, provider filters, CSV download')
    page.keyboard.press('/');page.locator('#search').fill('Understand gradients');page.locator('#search').press('Enter')
    page.wait_for_function("document.querySelector('h1')?.textContent === 'Understand gradients'");checks.append('Keyboard search navigation')
    go('metrics');page.screenshot(path=str(OUT/'desktop-metrics.png'),full_page=True)
    go('claims');page.screenshot(path=str(OUT/'desktop-claims.png'),full_page=True)
    go('start');page.screenshot(path=str(OUT/'desktop-start.png'),full_page=True)
    go('results');page.locator('.paper-result').last.wait_for();page.screenshot(path=str(OUT/'desktop-results.png'),full_page=True)
    page.locator('[data-figure="shape"]').screenshot(path=str(OUT/'paper-shape-desktop.png'))
    # Check every local download against the server and exported bytes.
    go('resources')
    for link in page.locator('a[download]').all():
        response=page.request.get('http://127.0.0.1:8765/'+link.get_attribute('href'));assert response.ok
        assert len(response.body())>0
    mobile=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1)
    mobile.on('pageerror',lambda e:errors.append(str(e)))
    for lesson in LESSONS:
        mobile.goto(f'http://127.0.0.1:8765/#/{lesson}');mobile.locator('h1').wait_for();mobile.wait_for_timeout(80)
        assert mobile.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'mobile {lesson}'
    mobile.goto('http://127.0.0.1:8765/#/results');mobile.locator('.paper-result').last.wait_for()
    mobile.locator('[data-figure="shape"]').screenshot(path=str(OUT/'paper-shape-mobile.png'))
    assert mobile.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert mobile.locator('[data-figure="shape"] img').evaluate('e=>e.currentSrc').endswith('shape-mobile.svg')
    checks.append('Paper figures use complete, readable plots sized for narrow screens')
    mobile.goto('http://127.0.0.1:8765/#/start');mobile.locator('h1').wait_for();mobile.locator('#menu-toggle').click();assert mobile.locator('#sidebar').evaluate("e=>e.classList.contains('open')")
    mobile.locator('.nav-link[href="#/eigen"]').click()
    mobile.wait_for_function("document.querySelector('h1')?.textContent === 'Build the eigen method'")
    assert mobile.locator('h1').inner_text()=='Build the eigen method'
    assert mobile.locator('#menu-toggle').get_attribute('aria-expanded')=='false'
    mobile.wait_for_timeout(300)
    mobile.screenshot(path=str(OUT/'mobile-eigen.png'),full_page=True)
    checks+=['All 12 routes at 390px without page overflow','Mobile menu navigation','Local downloads available']
    browser.close()
report={'checks':checks,'passed_checks':len(checks),'browser_errors':errors,'failed_local_requests':failed}
(OUT/'browser-report.json').write_text(json.dumps(report,indent=2)+'\n')
assert not errors,errors
assert not failed,failed
print(json.dumps(report,indent=2))
