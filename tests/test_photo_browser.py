"""Photo UI intake, confirmation and race regressions without private fixtures."""
from io import BytesIO
import json
from PIL import Image
import pytest
from test_web_e2e import server, browser, VIEWPORTS

pytestmark=pytest.mark.browser


def photo():
    out=BytesIO();Image.new('RGB',(320,240),'beige').save(out,'JPEG')
    return {'name':'photo.jpg','mimeType':'image/jpeg','buffer':out.getvalue()}


def draft(rank='A'):
    return {'ok':True,'board':['?']*18+[rank]+['?']*9,'waste':'K','trusted':True,'review':['20'],'note':'Check the stock counter.'}


def open_page(browser,server):
    page=browser.new_page(viewport=VIEWPORTS['phone']);page.goto(server);page.wait_for_selector('#editBoard .c');return page


def test_photo_bulk_confirm_keeps_unknowns_and_corrections(server,browser):
    page=open_page(browser,server)
    page.route('**/api/photo',lambda route:route.fulfill(json=draft()))
    page.click('#modeDeal');page.set_input_files('#photo',photo())
    page.wait_for_function("document.querySelector('#startBtn').textContent.includes('Confirm')")
    assert page.is_visible('#startBtn') and page.is_hidden('#solveBtn')
    assert page.locator('#editBoard .back').count()==27
    assert page.locator('#editBoard .review').count()==1
    page.click('#editBoard [data-p="20"]');page.click('#keys [data-k="2"]')
    assert page.locator('#editBoard .review').count()==0
    page.click('#startBtn');page.wait_for_selector('#board .c')
    assert page.locator('#board .back').count()==18
    assert page.locator('#board .ask').count()==8
    assert page.inner_text('#wst')=='K' and page.inner_text('#stk')=='24'
    page.close()


def test_stale_photo_does_not_replace_new_selection_or_edits(server,browser):
    page=open_page(browser,server);requests=[]
    page.route('**/api/photo',lambda route:requests.append(route))
    page.set_input_files('#photo',photo());page.wait_for_function("document.querySelector('#msg').textContent==='Reading the cards…'")
    page.wait_for_timeout(100)
    page.set_input_files('#photo',photo());page.wait_for_timeout(200)
    assert len(requests)==2
    requests[1].fulfill(json=draft('2'));page.wait_for_function("document.querySelector('#startBtn').textContent.includes('Confirm')")
    requests[0].fulfill(json=draft('A'));page.wait_for_timeout(100)
    assert page.locator('#editBoard [data-p="19"]').inner_text().endswith('2')
    page.set_input_files('#photo',photo());page.wait_for_timeout(200)
    page.click('#editBoard [data-p="19"]');page.click('#keys [data-k="3"]')
    requests[2].fulfill(json=draft('A'));page.wait_for_timeout(100)
    assert page.locator('#editBoard [data-p="19"]').inner_text().endswith('3')
    page.close()


def test_bad_photo_preserves_board_and_reselection_works(server,browser):
    page=open_page(browser,server)
    page.click('#wasteBtn');page.click('#keys [data-k="Q"]')
    page.set_input_files('#photo',{'name':'bad.jpg','mimeType':'image/jpeg','buffer':b'bad image'})
    page.wait_for_selector('#msg.err');assert 'JPEG' in page.inner_text('#msg')
    assert page.inner_text('#wasteBtn')=='Q'
    calls=[]
    def response(route):calls.append(1);route.fulfill(json=draft())
    page.route('**/api/photo',response)
    for i in range(2):
        page.set_input_files('#photo',photo());page.wait_for_function("document.querySelector('#photoNote').textContent.includes('Read in')")
        page.wait_for_timeout(100)
    assert len(calls)==2
    page.close()


def test_pasted_markup_never_enters_board(server,browser):
    page=open_page(browser,server);page.click('summary:has-text("Paste the board")')
    page.fill('#paste',' '.join(['?']*27+['<IMG/SRC/ONERROR=alert(1)>']));page.click('#pasteBtn')
    assert 'Use only' in page.inner_text('#msg')
    assert page.locator('#editBoard img').count()==0
    page.close()


def test_bitmap_decode_falls_back_to_image_element(server,browser):
    page=open_page(browser,server)
    page.evaluate("() => { window.createImageBitmap=async()=>{throw new Error('unsupported')}; }")
    page.route('**/api/photo',lambda route:route.fulfill(json=draft()))
    page.set_input_files('#photo',photo());page.wait_for_function("document.querySelector('#photoNote').textContent.includes('Read in')")
    assert page.locator('#msg.err').count()==0
    page.close()


def full_deal_draft():
    ranks = 'A 2 3 4 5 6 7 8 9 10 J Q K'.split() * 4
    return {
        'ok': True, 'mode': 'full_deal', 'trusted': True,
        'board': ranks[:28], 'waste': ranks[28],
        'stock': ranks[29:] + ['*'], 'stock_count': 24, 'joker': True,
        'review': [], 'note': 'Check all cards and the stock order.',
    }


def test_full_grid_photo_maps_editable_deal_and_bulk_solve(server, browser):
    page = open_page(browser, server)
    complete = full_deal_draft()
    uncertain = full_deal_draft()
    uncertain['board'][0] = '?'
    uncertain['waste'] = '?'
    uncertain['stock'][0] = '?'
    uncertain['review'] = ['1', 'waste', 'stock-1', 'stock-2']
    page.route('**/api/photo', lambda route: route.fulfill(json=uncertain))
    submitted = []
    def solve(route):
        submitted.append(route.request.post_data_json)
        route.fulfill(json={'ok': True, 'status': 'unsolvable', 'message': 'Deal checked.'})
    page.route('**/api/solve', solve)
    page.uncheck('#jk')
    page.set_input_files('#photo', photo())
    page.wait_for_function("document.querySelector('#solveBtn').textContent.includes('Confirm')")
    assert page.is_visible('#solveBtn') and page.is_hidden('#startBtn')
    assert page.is_visible('#stockOrder') and page.is_checked('#jk')
    assert page.inner_text('#stV') == '24'
    assert page.locator('#soChips .chip').count() == 24
    assert page.locator('#soChips .chip').nth(23).locator('span').inner_text() == '*'
    assert 'right to left' in page.inner_text('#gridMapping')
    assert page.locator('#editBoard .review').count() == 1
    assert page.locator('#wasteBtn.review').count() == 1
    assert page.locator('#soChips .review').count() == 2
    # Unknowns stay on the editor and cannot reach the solver.
    page.click('#solveBtn')
    error = page.inner_text('#msg')
    assert all(text in error for text in ('table positions 1', 'waste card', 'stock cards 1'))
    assert submitted == [] and page.is_visible('#setup')
    page.click('#editBoard [data-p="1"]'); page.click('#keys [data-k="A"]')
    page.click('#wasteBtn'); page.click('#keys [data-k="3"]')
    page.click('#soChips [data-i="0"]'); page.click('#keys [data-k="4"]')
    page.click('#soChips [data-i="1"]'); page.click('#keys [data-k="5"]')
    assert page.locator('.review').count() == 0
    page.click('#editBoard [data-p="1"]'); page.click('#keys [data-k="?"]')
    assert page.locator('#editBoard [data-p="1"].review').inner_text().endswith('?')
    page.click('#editBoard [data-p="1"]'); page.click('#keys [data-k="A"]')
    # A corrected stock can be marked unreadable again without removing its slot.
    page.click('#soChips [data-i="1"]'); page.click('#keys [data-k="?"]')
    page.click('#solveBtn')
    assert 'stock cards 2' in page.inner_text('#msg') and submitted == []
    page.click('#soChips [data-i="1"]'); page.click('#keys [data-k="5"]')
    page.click('#solveBtn'); page.wait_for_selector('#solved:not([hidden])')
    assert len(submitted) == 1
    assert submitted[0]['board'] == ' '.join(complete['board'])
    assert submitted[0]['waste'] == complete['waste']
    assert submitted[0]['stock'] == complete['stock']
    assert submitted[0]['stock_count'] == 24
    page.close()


@pytest.mark.parametrize('failure', ['decode', 'reader', 'malformed_stock'])
def test_failed_grid_replacement_preserves_complete_draft(server, browser, failure):
    import base64
    page = open_page(browser, server)
    previous = full_deal_draft()
    previous['overlay'] = 'data:image/jpeg;base64,' + base64.b64encode(photo()['buffer']).decode()
    previous['review'] = ['stock-3']
    responses = [previous]
    if failure == 'reader':
        responses.append({'ok': False, 'message': 'Could not locate the grid.'})
    elif failure == 'malformed_stock':
        replacement = full_deal_draft()
        replacement['board'][0] = 'K'
        replacement['stock'].pop()
        responses.append(replacement)
    page.route('**/api/photo', lambda route: route.fulfill(json=responses.pop(0)))
    page.set_input_files('#photo', photo())
    page.wait_for_function("document.querySelector('#solveBtn').textContent.includes('Confirm')")
    before = page.evaluate("({setup, note:document.querySelector('#photoNote').textContent, review:[...photoReview]})")
    page.set_input_files('#photo', {'name': 'bad.jpg', 'mimeType': 'image/jpeg', 'buffer': b'bad'} if failure == 'decode' else photo())
    page.wait_for_selector('#msg.err')
    after = page.evaluate("({setup, note:document.querySelector('#photoNote').textContent, review:[...photoReview]})")
    assert after == before
    assert page.is_visible('#solveBtn') and page.is_visible('#photoView')
    assert page.locator('#photoView').get_attribute('src') == previous['overlay']
    page.close()


def test_tableau_after_grid_clears_known_stock_and_mode(server, browser):
    page = open_page(browser, server)
    responses = [full_deal_draft(), draft()]
    page.route('**/api/photo', lambda route: route.fulfill(json=responses.pop(0)))
    page.set_input_files('#photo', photo())
    page.wait_for_function("document.querySelector('#solveBtn').textContent.includes('Confirm')")
    page.set_input_files('#photo', photo())
    page.wait_for_selector('#startBtn:not([hidden])')
    assert page.is_hidden('#stockOrder') and page.is_hidden('#gridMapping')
    page.click('#modeDeal')
    assert page.locator('#soChips .chip').count() == 0
    assert page.is_hidden('#gridMapping')
    page.close()


def test_stale_grid_response_cannot_change_new_photo_mode_or_stock_edit(server, browser):
    page = open_page(browser, server)
    requests = []
    page.route('**/api/photo', lambda route: requests.append(route))
    def choose(count):
        # Wait for the request itself; a slow runner may take longer than a fixed pause to encode.
        page.set_input_files('#photo', photo())
        for _ in range(200):
            if len(requests) >= count: break
            page.wait_for_timeout(50)
    choose(1); choose(2)
    assert len(requests) == 2
    requests[1].fulfill(json=draft('2'))
    page.wait_for_function("document.querySelector('#startBtn').textContent.includes('Confirm')")
    requests[0].fulfill(json=full_deal_draft()); page.wait_for_timeout(100)
    assert page.is_visible('#startBtn') and page.is_hidden('#solveBtn')
    assert page.locator('#soChips .chip').count() == 0
    choose(3)
    requests[2].fulfill(json=full_deal_draft())
    page.wait_for_selector('#solveBtn:not([hidden])')
    choose(4)
    page.click('#soChips [data-i="0"]'); page.click('#keys [data-k="?"]')
    requests[3].fulfill(json=draft()); page.wait_for_timeout(100)
    assert page.is_visible('#solveBtn') and page.is_hidden('#startBtn')
    assert page.locator('#soChips [data-i="0"] span').inner_text() == '?'
    assert page.locator('#soChips .chip').count() == 24
    page.close()
