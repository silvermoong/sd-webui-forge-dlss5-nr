import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const origin = 'http://127.0.0.1:7875';
const folder = path.join(root, 'work', `ui-browser-${Date.now()}`);
await mkdir(folder, {recursive: true});
const report = {folder, errors: [], viewports: [], languages: [], multipass: [], passViewports: [], img2img: [], img2imgViewports: [], direct: [], directViewports: []};
const browser = await chromium.launch({channel: 'msedge', headless: true,
  args: ['--disable-gpu', '--disable-background-networking', '--disable-component-update', '--disable-sync']});
const context = await browser.newContext({viewport: {width: 1440, height: 1000}, serviceWorkers: 'block'});
await context.route('**/*', route => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
const page = await context.newPage();
page.on('pageerror', error => report.errors.push(String(error)));
report.presetRequests = [];
page.on('request', request => {
  if (request.method() !== 'POST' || !request.url().includes('/queue/join')) return;
  const data = request.postDataJSON();
  const selection = data?.data?.[0];
  if (typeof selection === 'string' && selection.startsWith('{"name":')) {
    report.presetRequests.push({selection: JSON.parse(selection), fn: data.fn_index});
  }
});

async function selectPreset(prefix, name) {
  await page.locator(`#${prefix}_preset_list input`).click();
  await page.locator(`#${prefix}_preset_list`).getByRole('option', {name, exact: true}).click();
}

async function presetToolbar(prefix, locale) {
  await page.waitForFunction(id => {
    const buttons = [...document.querySelectorAll(`#${id} button.forge-nr-preset-button`)];
    return buttons.length === 2 && buttons.every(button => {
      const icon = button.querySelector('img');
      return icon?.complete && icon.naturalWidth > 0;
    });
  }, `${prefix}_preset_toolbar`);
  const geometry = await page.locator(`#${prefix}_preset_toolbar`).evaluate(element => ({
    top: element.getBoundingClientRect().top,
    feedbackClipped: (() => {
      const container = document.getElementById(element.id.replace('_preset_toolbar', '_preset_message'));
      if (!container?.getClientRects().length) return false;
      const content = container.querySelector('textarea') || container;
      return content.scrollHeight > content.clientHeight + 1;
    })(),
    textRight: (() => {
      const input = element.querySelector('input[role="listbox"]');
      return input.getBoundingClientRect().right - parseFloat(getComputedStyle(input).paddingRight);
    })(),
    arrowLeft: element.querySelector('svg.dropdown-arrow').getBoundingClientRect().left,
    buttons: [...element.querySelectorAll('button.forge-nr-preset-button')].map(button => ({
      title: button.title, label: button.getAttribute('aria-label'),
      width: button.getBoundingClientRect().width, height: button.getBoundingClientRect().height,
      left: button.getBoundingClientRect().left, right: button.getBoundingClientRect().right,
      top: button.getBoundingClientRect().top,
      embeddedIcon: button.querySelector('img').currentSrc.startsWith('data:image/svg+xml;base64,'),
      ink: (() => {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 24;
        const context = canvas.getContext('2d');
        context.drawImage(button.querySelector('img'), 0, 0, 24, 24);
        return context.getImageData(0, 0, 24, 24).data.filter((value, index) => index % 4 === 3 && value > 0).length;
      })()
    }))
  }));
  const labels = locale === 'zh' ? ['保存/覆盖当前参数', '删除选中预设'] :
    ['Save / replace preset', 'Delete selected preset'];
  assert.deepEqual(geometry.buttons.map(button => button.title), labels);
  assert.deepEqual(geometry.buttons.map(button => button.label), labels);
  assert.equal(geometry.feedbackClipped, false, 'Preset feedback must remain fully visible');
  assert.equal(await page.locator(`#${prefix}_preset_list input`).getAttribute('placeholder'), locale === 'zh' ? '预设' : 'Preset');
  assert.ok(geometry.textRight <= geometry.arrowLeft + 1, 'Preset name must not overlap the dropdown arrow');
  assert.ok(geometry.top < (await page.locator(`#${prefix}_passes`).boundingBox()).y);
  assert.equal(await page.locator(`#${prefix}_preset_name`).count(), 0);
  for (const button of geometry.buttons) {
    assert.equal(button.embeddedIcon, true, 'Preset icons must not depend on Forge startup file caching');
    assert.ok(button.ink > 20, 'Preset icon pixels must not be blank');
    assert.ok(Math.abs(button.width - 32) <= 1 && Math.abs(button.height - 32) <= 1, 'Preset actions must remain square');
    assert.ok(Math.abs(button.top - geometry.buttons[0].top) <= 1, 'Preset actions must stay on one row');
    assert.ok(button.left >= -1 && button.right <= page.viewportSize().width + 1);
  }
}

async function presetCrud(locale) {
  const prefix = 'forge_nr_direct';
  const field = page.locator(`#${prefix}_preset_list input`);
  const name = `toolbar-copy-${locale}-${Date.now()}`;
  const state = page.locator(`#${prefix}_preset_state span`);
  const catalog = page.locator(`#${prefix}_preset_catalog textarea`);
  const selection = page.locator(`#${prefix}_preset_selection textarea`);
  const message = async pattern => page.waitForFunction(expected =>
    new RegExp(expected).test(document.querySelector('#forge_nr_direct_preset_message')?.textContent || ''), pattern);
  const savedState = locale === 'zh' ? '已保存' : 'Saved';
  const dirtyState = locale === 'zh' ? '已修改 · 未保存' : 'Modified (not saved)';
  const waitState = expected => page.waitForFunction(text =>
    document.querySelector('#forge_nr_direct_preset_state span')?.textContent === text, expected);
  const confirm = async (action, accept) => {
    const requested = page.waitForEvent('dialog', {timeout: 5000});
    const click = page.locator(`#${prefix}_${action}_preset`).click();
    const dialog = await requested;
    assert.equal(dialog.type(), 'confirm');
    assert.ok(dialog.message().includes(name));
    if (accept) await dialog.accept(); else await dialog.dismiss();
    await click;
  };
  const select = async title => {
    await field.click();
    await page.locator(`#${prefix}_preset_list`).getByRole('option', {name: title, exact: true}).click();
  };
  await page.locator('#forge_nr_direct').getByRole('tab', {name: '1st', exact: true}).click();
  const mix = page.locator(`#${prefix}_mix input[type="number"]`);
  await select('hires-fixture');
  await page.waitForFunction(() => Number(document.querySelector('#forge_nr_direct_mix input[type="number"]')?.value) === .31);
  await mix.fill('.91');
  await mix.press('Tab');
  await select('hires-fixture');
  await page.waitForFunction(() => Number(document.querySelector('#forge_nr_direct_mix input[type="number"]')?.value) === .31,
    null, {timeout: 4000});
  const priorSelection = await selection.inputValue();
  await field.fill('legacy-fixture');
  await field.press('Enter');
  assert.equal(await selection.inputValue(), priorSelection, 'Typing an existing name must not load it');
  assert.equal(Number(await mix.inputValue()), .31);
  await field.click();
  await field.fill('legacy-fixture');
  await field.press('ArrowDown');
  assert.equal(await selection.inputValue(), priorSelection, 'Arrow navigation must not apply parameters');
  await field.press('Escape');
  await field.press('Tab');
  assert.equal(await selection.inputValue(), priorSelection, 'Escape and blur must not apply parameters');
  await field.click();
  await field.fill('legacy-fixture');
  const options = page.locator(`#${prefix}_preset_list [role="option"]`);
  await options.first().waitFor({state: 'visible'});
  const optionCount = await options.count();
  for (let attempt = 0; attempt < optionCount; attempt++) {
    await field.press('ArrowDown');
    if (await page.locator(`#${prefix}_preset_list [role="option"].active`).getAttribute('aria-label') === 'legacy-fixture') break;
  }
  assert.equal(await page.locator(`#${prefix}_preset_list [role="option"].active`).getAttribute('aria-label'), 'legacy-fixture');
  await field.press('Enter');
  await page.waitForFunction(() => Number(document.querySelector('#forge_nr_direct_mix input[type="number"]')?.value) === .25);
  await field.fill(name);
  await page.locator(`#${prefix}_save_preset`).click();
  await message('预设已保存|Preset saved');
  await waitState(savedState);
  assert.equal(await field.inputValue(), name);
  await select(name);
  await message('预设已载入|Preset loaded');
  await mix.fill('.27');
  await mix.press('Tab');
  await waitState(dirtyState);
  await mix.fill('.25');
  await mix.press('Tab');
  await waitState(savedState);
  await mix.fill('.27');
  await mix.press('Tab');
  await waitState(dirtyState);
  const beforeReplace = await catalog.inputValue();
  await confirm('save', false);
  await message('预设未改动|Presets unchanged');
  assert.equal(await catalog.inputValue(), beforeReplace);
  assert.equal(Number(await mix.inputValue()), .27);
  assert.equal(await state.textContent(), dirtyState);
  await confirm('save', true);
  await message('预设已保存|Preset saved');
  await waitState(savedState);
  assert.equal(JSON.parse(await catalog.inputValue())[name][0].params.mix, .27);
  await mix.fill('.91');
  await mix.press('Tab');
  await waitState(dirtyState);
  await select(name);
  await page.waitForFunction(() => Number(document.querySelector('#forge_nr_direct_mix input[type="number"]')?.value) === .27);
  await waitState(savedState);
  const beforeDelete = await catalog.inputValue();
  await confirm('delete', false);
  await message('预设未改动|Presets unchanged');
  assert.equal(await catalog.inputValue(), beforeDelete);
  assert.equal(await field.inputValue(), name);
  assert.equal(Number(await mix.inputValue()), .27);
  await confirm('delete', true);
  await message('预设已删除|Preset deleted');
  await waitState('');
  assert.equal(await field.inputValue(), '');
  assert.equal(Number(await mix.inputValue()), .27);
  await field.click();
  assert.equal(await page.getByRole('option', {name, exact: true}).count(), 0);
  await page.getByRole('option', {name: 'hires-fixture', exact: true}).click();
  await page.waitForFunction(() => Number(document.querySelector('#forge_nr_direct_mix input[type="number"]')?.value) === .31);
}

async function capture(enabled, img2img = false) {
  const suffix = img2img ? '_img2img' : '';
  const selector = `#fixture_request${suffix} textarea`;
  const previous = await page.locator(selector).inputValue();
  const sequence = previous ? JSON.parse(previous).sequence : 0;
  await page.locator(`#fixture_capture${suffix}`).click();
  await page.waitForFunction(expected => {
    try {
      const value = JSON.parse(document.querySelector(expected.selector).value);
      return value.sequence > expected.sequence && value.args[0] === expected.enabled;
    } catch { return false; }
  }, {sequence, enabled, selector});
  const result = JSON.parse(await page.locator(selector).inputValue());
  assert.equal(result.synthetic, true);
  assert.equal(result.args.length, 35);
  return result.args;
}

async function multipass(locale) {
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto(`${origin}/${locale}/`, {waitUntil: 'load'});
  const master = page.locator('#forge_nr-visible-checkbox');
  await master.waitFor({state: 'visible'});
  await page.waitForFunction(() => document.querySelector('#forge_nr_runtime textarea')?.value.includes('runtime_id'));
  await page.locator('#forge_nr > .label-wrap').click();
  await master.uncheck();
  await presetToolbar('forge_nr', locale);
  const tab = index => page.getByRole('tab', {name: ['1st', '2nd', '3rd'][index - 1], exact: true}).click();
  const enabled = index => page.locator(`#forge_nr_pass_${index}_enabled input[type="checkbox"]`);
  const key = (index, name) => `forge_nr_${index === 1 ? '' : `pass_${index}_`}${name}`;
  const snapshot = async () => {
    await capture(true);
    return JSON.parse(await page.locator('#fixture_request textarea').inputValue());
  };
  const number = async (index, name, value) => {
    const field = page.locator(`#${key(index, name)} input[type="number"]`);
    await field.fill(String(value));
    await field.press('Tab');
  };
  const stage = async (index, after) => {
    const label = locale === 'zh' ? (after ? '高清修复后' : '高清修复前（未开高清：首轮后）') :
      (after ? 'After Hires. fix' : 'Before Hires. fix (after first pass if Hires is off)');
    await page.locator(`#${key(index, 'stage')}`).getByRole('radio', {name: label, exact: true}).check();
  };
  assert.equal(await enabled(1).isChecked(), true);
  await tab(2);
  assert.equal(await enabled(2).isChecked(), false);
  await tab(3);
  assert.equal(await enabled(3).isChecked(), false);
  await tab(1);
  await master.check();
  const initial = await snapshot();
  assert.equal(initial.spec.params.mix, 1);
  await page.locator('#fixture_hr input[type="checkbox"]').check();
  const stages = [key(1, 'stage'), key(2, 'stage'), key(3, 'stage')];
  await page.waitForFunction(ids => ids.every(id =>
    document.getElementById(id)?.querySelectorAll('input[type="radio"]').length === 2), stages);
  await number(1, 'tone', .71);
  await number(1, 'mix', .55);
  await stage(1, true);
  await tab(2);
  await page.locator('#forge_nr_copy_pass_2').click();
  await page.waitForFunction(() => Number(document.querySelector('#forge_nr_pass_2_mix input[type="number"]')?.value) === .55);
  assert.equal(await enabled(2).isChecked(), false);
  assert.equal(Number(await page.locator('#forge_nr_pass_2_tone input[type="number"]').inputValue()), .71);
  await enabled(2).check();
  await number(2, 'tone', .83);
  await number(2, 'mix', .35);
  await page.locator('#forge_nr_pass_2_style input').click();
  await page.getByRole('option', {name: '2', exact: true}).click();
  await stage(2, false);
  await tab(3);
  assert.equal(Number(await page.locator('#forge_nr_pass_3_mix input[type="number"]').inputValue()), 1);
  await enabled(3).check();
  await number(3, 'tone', 1.17);
  await number(3, 'mix', .72);
  await stage(3, true);
  let record = await snapshot();
  assert.deepEqual(record.execution_order.map(pass => pass.index), [2, 1, 3]);
  assert.deepEqual(record.spec.passes.map(pass => pass.params.tone), [.71, .83, 1.17]);
  assert.deepEqual(record.spec.passes.map(pass => pass.params.mix), [.55, .35, .72]);
  await tab(1);
  await number(1, 'tone', .44);
  record = await snapshot();
  assert.deepEqual(record.spec.passes.map(pass => pass.params.tone), [.44, .83, 1.17]);
  await tab(2);
  await enabled(2).uncheck();
  record = await snapshot();
  assert.deepEqual(record.execution_order.map(pass => pass.index), [1, 3]);
  assert.equal(record.spec.passes[1].params.style, 2);
  await enabled(2).check();
  const saved = await snapshot();
  const presetName = `three-pages-${locale}-${Date.now()}`;
  await page.locator('#forge_nr_preset_list input').fill(presetName);
  await page.locator('#forge_nr_preset_list input').press('Enter');
  await page.locator('#forge_nr_save_preset').click();
  await page.waitForFunction(() => /预设已保存|Preset saved/.test(
    document.querySelector('#forge_nr_preset_message')?.textContent || ''));
  await tab(3);
  await number(3, 'mix', .1);
  await enabled(3).uncheck();
  await master.uncheck();
  await selectPreset('forge_nr', presetName);
  await page.waitForFunction(() => document.querySelector('#forge_nr_pass_3_enabled input')?.checked === true);
  assert.equal(await master.isChecked(), false);
  await capture(false);
  assert.equal(JSON.parse(await page.locator('#fixture_request textarea').inputValue()).spec, null);
  await master.check();
  record = await snapshot();
  assert.deepEqual(record.spec, saved.spec);
  assert.equal(record.args[11], initial.args[11]);
  for (const index of [1, 2, 3]) {
    await tab(index);
    await enabled(index).uncheck();
  }
  assert.equal((await snapshot()).spec, null);
  await selectPreset('forge_nr', presetName);
  await page.waitForFunction(() => [1, 2, 3].every(index =>
    document.querySelector(`#forge_nr_pass_${index}_enabled input`)?.checked === true));
  assert.deepEqual((await snapshot()).spec, saved.spec);
  for (const [width, height] of [[1440,1000], [1040,900], [768,900], [390,844], [320,720], [1440,540]]) {
    await page.setViewportSize({width, height});
    for (const index of [1, 2, 3]) {
      await tab(index);
      await presetToolbar('forge_nr', locale);
      const panel = page.locator('#forge_nr_passes');
      await panel.screenshot({path: path.join(folder, `passes-${locale}-${width}x${height}-${index}.png`), animations: 'disabled'});
      const geometry = await page.evaluate(() => {
        const fields = [...document.querySelectorAll('#forge_nr_passes input, #forge_nr_passes button')]
          .filter(element => element.getClientRects().length && element.getBoundingClientRect().width > 0);
        return {width: innerWidth, height: innerHeight, documentWidth: document.documentElement.scrollWidth,
          overflow: fields.map(element => ({id: element.closest('[id]')?.id,
            left: element.getBoundingClientRect().left, right: element.getBoundingClientRect().right}))
            .filter(bounds => bounds.left < -1 || bounds.right > innerWidth + 1),
          clippedButtons: fields.filter(element => element.tagName === 'BUTTON' && element.scrollWidth > element.clientWidth + 1)
            .map(element => element.textContent)};
      });
      assert.ok(geometry.documentWidth <= geometry.width + 1, `Page overflow at ${locale}/${width}`);
      assert.deepEqual(geometry.overflow, [], `Controls overflow at ${locale}/${width}/tab${index}`);
      assert.deepEqual(geometry.clippedButtons, [], `Clipped buttons at ${locale}/${width}/tab${index}`);
      report.passViewports.push({locale, tab: index, ...geometry});
    }
  }
  await page.locator('#fixture_hr input[type="checkbox"]').uncheck();
  await page.waitForFunction(ids => ids.every(id =>
    document.getElementById(id)?.querySelectorAll('input[type="radio"]').length === 1), stages);
  record = await snapshot();
  assert.deepEqual(record.spec.passes.map(pass => pass.stage), ['before_hr', 'before_hr', 'before_hr']);
  assert.deepEqual(record.execution_order.map(pass => pass.index), [1, 2, 3]);
  await page.locator('#forge_nr_preset_list input').click();
  await page.getByRole('option', {name: 'legacy-fixture', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_pass_3_enabled input')?.checked === false);
  record = await snapshot();
  assert.equal(record.spec.params.mix, .25);
  assert.equal(record.spec.stage, 'before_hr');
  assert.equal(record.execution_order.length, 1);
  assert.equal(record.args[11], initial.args[11]);
  report.multipass.push({locale, independentParams: true, disabledSkip: true, allDisabled: true,
    stageOrder: [2, 1, 3], copyKeepsDisabled: true, presets: true, legacyMigration: true});
}

async function img2img(locale) {
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto(`${origin}/i2i-${locale}/`, {waitUntil: 'load'});
  await page.waitForFunction(() => ['forge_nr', 'forge_nr_img2img'].every(prefix =>
    document.querySelector(`#${prefix}_runtime textarea`)?.value.includes('runtime_id')));
  const duplicates = await page.evaluate(() => {
    const ids = [...document.querySelectorAll('[id^="forge_nr"]')].map(element => element.id);
    return ids.filter((id, index) => ids.indexOf(id) !== index);
  });
  assert.deepEqual(duplicates, []);
  await page.locator('#forge_nr > .label-wrap').click();
  await page.locator('#forge_nr-visible-checkbox').uncheck();
  await page.locator('#forge_nr_mix input[type="number"]').fill('.19');
  await page.locator('#forge_nr_mix input[type="number"]').press('Tab');
  await page.locator('#fixture_hr input').check();
  const txtArgs = await capture(false);
  const initial = JSON.parse(await page.locator('#fixture_request textarea').inputValue());
  await page.getByRole('tab', {name: 'img2img', exact: true}).click();
  const master = page.locator('#forge_nr_img2img-visible-checkbox');
  assert.equal(await master.isChecked(), false);
  await page.locator('#forge_nr_img2img > .label-wrap').click();
  await master.uncheck();
  await presetToolbar('forge_nr_img2img', locale);
  const imgArgs = await capture(false, true);
  assert.equal(imgArgs[9], 1);
  assert.equal(txtArgs[9], .19);
  const panel = page.locator('#forge_nr_img2img');
  const tab = index => panel.getByRole('tab', {name: ['1st', '2nd', '3rd'][index - 1], exact: true}).click();
  const enabled = index => page.locator(`#forge_nr_img2img_pass_${index}_enabled input[type="checkbox"]`);
  const snapshot = async (on = true) => {
    await capture(on, true);
    return JSON.parse(await page.locator('#fixture_request_img2img textarea').inputValue());
  };
  await page.locator('#forge_nr_img2img_preset_list input').click();
  await page.getByRole('option', {name: 'hires-fixture', exact: true}).click();
  await page.waitForFunction(() => Number(document.querySelector('#forge_nr_img2img_pass_3_mix input[type="number"]')?.value) === .73);
  assert.equal(await master.isChecked(), false);
  assert.match(await page.locator('#forge_nr_img2img_preset_message').innerText(), /图生图后|after img2img/);
  await master.check();
  let record = await snapshot();
  assert.deepEqual(record.spec.passes.map(pass => pass.params.mix), [.31, .52, .73]);
  assert.deepEqual(record.spec.passes.map(pass => pass.stage), ['before_hr', 'before_hr', 'before_hr']);
  assert.deepEqual(record.execution_order.map(pass => pass.index), [1, 2, 3]);
  assert.deepEqual(record.saved_presets, initial.saved_presets);
  assert.equal(record.args[11], imgArgs[11]);
  for (const index of [1, 2, 3]) {
    await tab(index);
    const prefix = index === 1 ? '' : `pass_${index}_`;
    const stage = page.locator(`#forge_nr_img2img_${prefix}stage`);
    assert.equal(await stage.locator('input[type="radio"]').count(), 1);
    assert.equal(await stage.locator('input[type="radio"]').evaluate(element => element.disabled), true);
    assert.match(await stage.innerText(), /图生图后|After img2img/);
    assert.doesNotMatch(await stage.innerText(), /高清|Hires/);
  }
  await tab(2);
  await enabled(2).uncheck();
  assert.deepEqual((await snapshot()).execution_order.map(pass => pass.index), [1, 3]);
  await page.locator('#forge_nr_img2img_copy_pass_2').click();
  await page.waitForFunction(() => Number(document.querySelector('#forge_nr_img2img_pass_2_mix input[type="number"]')?.value) === .31);
  assert.equal(await enabled(2).isChecked(), false);
  for (const index of [1, 3]) {
    await tab(index);
    await enabled(index).uncheck();
  }
  assert.equal((await snapshot()).spec, null);
  await selectPreset('forge_nr_img2img', 'hires-fixture');
  await page.waitForFunction(() => [1, 2, 3].every(index =>
    document.querySelector(`#forge_nr_img2img_pass_${index}_enabled input`)?.checked));
  record = await snapshot();
  assert.deepEqual(record.saved_presets, initial.saved_presets);
  for (const [width, height] of [[1440,1000], [1040,900], [768,900], [390,844], [320,720], [1440,540]]) {
    await page.setViewportSize({width, height});
    for (const index of [1, 2, 3]) {
      await tab(index);
      await presetToolbar('forge_nr_img2img', locale);
      await page.locator('#forge_nr_img2img_passes').screenshot({
        path: path.join(folder, `img2img-${locale}-${width}x${height}-${index}.png`), animations: 'disabled'});
      const geometry = await page.evaluate(() => {
        const fields = [...document.querySelectorAll('#forge_nr_img2img_passes input, #forge_nr_img2img_passes button')]
          .filter(element => element.getClientRects().length && element.getBoundingClientRect().width > 0);
        return {width: innerWidth, height: innerHeight, documentWidth: document.documentElement.scrollWidth,
          overflow: fields.filter(element => element.getBoundingClientRect().left < -1 || element.getBoundingClientRect().right > innerWidth + 1)
            .map(element => element.closest('[id]')?.id),
          clippedButtons: fields.filter(element => element.tagName === 'BUTTON' && element.scrollWidth > element.clientWidth + 1)
            .map(element => element.textContent)};
      });
      assert.ok(geometry.documentWidth <= geometry.width + 1);
      assert.deepEqual(geometry.overflow, []);
      assert.deepEqual(geometry.clippedButtons, []);
      report.img2imgViewports.push({locale, tab: index, ...geometry});
    }
  }
  await page.setViewportSize({width: 1440, height: 1000});
  await page.getByRole('tab', {name: 'txt2img', exact: true}).click();
  assert.deepEqual(await capture(false), txtArgs);
  await page.locator('#fixture_hr input').uncheck();
  await page.getByRole('tab', {name: 'img2img', exact: true}).click();
  assert.deepEqual((await snapshot()).spec, record.spec);
  report.img2img.push({locale, independentArguments: true, fixedStage: true, presetFileUnchanged: true,
    stageOrder: [1, 2, 3], disabledSkip: true, allDisabled: true, copyKeepsDisabled: true});
}

function pngReceipt(bytes) {
  assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  let offset = 8;
  const text = {};
  while (offset < bytes.length) {
    const length = bytes.readUInt32BE(offset);
    assert.ok(offset + length + 12 <= bytes.length);
    if (bytes.toString('ascii', offset + 4, offset + 8) === 'tEXt') {
      const data = bytes.subarray(offset + 8, offset + 8 + length);
      const delimiter = data.indexOf(0);
      assert.ok(delimiter > 0);
      text[data.toString('latin1', 0, delimiter)] = data.toString('latin1', delimiter + 1);
    }
    offset += length + 12;
  }
  return JSON.parse(text['DLSS5 NR']);
}

async function direct(locale) {
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto(`${origin}/direct-${locale}/`, {waitUntil: 'load'});
  await page.waitForFunction(() => ['forge_nr', 'forge_nr_img2img', 'forge_nr_direct'].every(prefix =>
    document.querySelector(`#${prefix}_runtime textarea`)?.value.includes('runtime_id')));
  const txtArgs = await capture(false);
  await page.getByRole('tab', {name: 'img2img', exact: true}).click();
  const imgArgs = await capture(false, true);
  await page.getByRole('tab', {name: 'DLSS5 NR', exact: true}).click();
  const settings = page.locator('#forge_nr_direct');
  await presetToolbar('forge_nr_direct', locale);
  assert.equal(await page.locator('#forge_nr_direct_preset_message').isVisible(), false);
  const run = page.locator('#forge_nr_direct_enhance');
  const cancel = page.locator('#forge_nr_direct_cancel');
  const status = page.locator('#forge_nr_direct_status textarea');
  const sourceArea = page.locator('#forge_nr_direct_source');
  assert.equal(await page.locator('#forge_nr_direct_original').isVisible(), false);
  const emptySource = await sourceArea.boundingBox();
  assert.equal(Math.round(emptySource.height), 360);
  await page.locator('#fixture_direct_tab').screenshot({path: path.join(folder, `direct-${locale}-empty.png`), animations: 'disabled'});
  assert.equal(await page.locator('#forge_nr_direct-checkbox').count(), 0);
  assert.equal(await cancel.isDisabled(), true);
  await run.click();
  await page.waitForFunction(() => /先上传|Upload a still image/.test(document.querySelector('#forge_nr_direct_status textarea')?.value || ''));
  const sourceBytes = Buffer.from(await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 96;
    const drawing = canvas.getContext('2d');
    const pixels = drawing.createImageData(canvas.width, canvas.height);
    for (let row = 0; row < canvas.height; row++) {
      for (let column = 0; column < canvas.width; column++) {
        const offset = (row * canvas.width + column) * 4;
        pixels.data.set([30 + column % 180, 60 + row * 2, 210 - column, column > 110 ? 128 : 255], offset);
      }
    }
    drawing.putImageData(pixels, 0, 0);
    return canvas.toDataURL('image/png').split(',')[1];
  }), 'base64');
  await page.locator('#forge_nr_direct_input input[type="file"]').setInputFiles({name: 'direct-fixture.png', mimeType: 'image/png', buffer: sourceBytes});
  await page.waitForFunction(() => document.querySelector('#forge_nr_direct_original img')?.naturalWidth === 160);
  assert.equal(await sourceArea.locator('img').count(), 1);
  const loadedSource = await sourceArea.boundingBox();
  assert.ok(Math.abs(loadedSource.height - emptySource.height) <= 1, 'Uploading must not resize the original-image area');
  assert.ok(Math.abs((await page.locator('#forge_nr_direct_input').boundingBox()).height - 64) <= 1);
  const sourceUrl = await page.locator('#forge_nr_direct_original img').getAttribute('src');
  const succeed = async () => {
    const existing = page.locator('#forge_nr_direct_output img');
    const previous = await existing.count() ? await existing.getAttribute('src') : null;
    await run.click();
    await page.waitForFunction(old => {
      const image = document.querySelector('#forge_nr_direct_output img');
      return image?.naturalWidth === 160 && image.getAttribute('src') !== old &&
        /已完成|Completed/.test(document.querySelector('#forge_nr_direct_status textarea')?.value || '');
    }, previous);
    assert.equal(await run.isDisabled(), false);
    assert.equal(await cancel.isDisabled(), true);
    const url = await page.locator('#forge_nr_direct_output img').getAttribute('src');
    assert.equal(new URL(url, origin).origin, origin);
    const response = await context.request.get(new URL(url, origin).href);
    assert.equal(response.ok(), true);
    const receipt = pngReceipt(await response.body());
    assert.equal(receipt.mode, 'direct');
    assert.equal(receipt.status, 'done');
    assert.equal(receipt.count, 1);
    assert.deepEqual([receipt.width, receipt.height], [160, 96]);
    assert.match(receipt.request_id, /^[a-f0-9]{32}$/);
    return receipt;
  };
  const first = await succeed();
  assert.equal(first.pass_count, 1);
  const pixels = await page.evaluate(async () => {
    const read = selector => {
      const image = document.querySelector(selector);
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const drawing = canvas.getContext('2d');
      drawing.drawImage(image, 0, 0);
      return [...drawing.getImageData(0, 0, canvas.width, canvas.height).data];
    };
    return {source: read('#forge_nr_direct_original img'), result: read('#forge_nr_direct_output img')};
  });
  let changed = 0;
  for (let index = 0; index < pixels.source.length; index += 4) {
    assert.equal(pixels.result[index + 3], pixels.source[index + 3]);
    if (pixels.source[index + 3] === 255) {
      for (let channel = 0; channel < 3; channel++) {
        assert.equal(pixels.result[index + channel], pixels.source[index + channel] >= 128 ? 224 : 16);
        if (pixels.result[index + channel] !== pixels.source[index + channel]) changed++;
      }
    }
  }
  assert.ok(changed > 1000);
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#forge_nr_direct_output a[download]').click();
  const download = await downloadPromise;
  const saved = path.join(folder, `direct-${locale}-download.png`);
  await download.saveAs(saved);
  assert.deepEqual(pngReceipt(await readFile(saved)), first);
  const repeated = await succeed();
  assert.notEqual(repeated.request_id, first.request_id);
  assert.equal(repeated.input_sha256, first.input_sha256);
  await page.locator('#forge_nr_direct_preset_list input').click();
  await page.getByRole('option', {name: 'hires-fixture', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_direct_pass_3_enabled input')?.checked === true);
  assert.match(await page.locator('#forge_nr_direct_preset_message').innerText(), /上传图片|uploaded image/);
  const multiple = await succeed();
  assert.deepEqual(multiple.passes.map(pass => pass.index), [1, 2, 3]);
  assert.equal(multiple.pass_count, 3);
  await presetCrud(locale);
  const tab = index => settings.getByRole('tab', {name: ['1st', '2nd', '3rd'][index - 1], exact: true}).click();
  for (const [width, height] of [[1440,1000], [1040,900], [768,900], [390,844], [320,720], [1440,540]]) {
    await page.setViewportSize({width, height});
    for (const index of [1, 2, 3]) {
      await tab(index);
      await presetToolbar('forge_nr_direct', locale);
      await page.locator('#fixture_direct_tab').screenshot({path: path.join(folder, `direct-${locale}-${width}x${height}-${index}.png`), animations: 'disabled'});
      const geometry = await page.evaluate(() => {
        const fields = [...document.querySelectorAll('#fixture_direct_tab input, #fixture_direct_tab button, #fixture_direct_tab textarea')]
          .filter(element => element.getClientRects().length && element.getBoundingClientRect().width > 0);
        return {width: innerWidth, height: innerHeight, documentWidth: document.documentElement.scrollWidth,
          overflow: fields.filter(element => element.getBoundingClientRect().left < -1 || element.getBoundingClientRect().right > innerWidth + 1)
            .map(element => element.closest('[id]')?.id),
          clippedButtons: fields.filter(element => {
            if (element.tagName !== 'BUTTON') return false;
            const label = element.classList.contains('label-wrap') ? element.firstElementChild : element;
            const bounds = element.getBoundingClientRect();
            const textBounds = label.getBoundingClientRect();
            return label.scrollWidth > label.clientWidth + 1 || textBounds.left < bounds.left - 1 || textBounds.right > bounds.right + 1;
          })
            .map(element => element.textContent)};
      });
      assert.ok(geometry.documentWidth <= geometry.width + 1, `Direct page overflow ${locale}/${width}`);
      assert.deepEqual(geometry.overflow, []);
      assert.deepEqual(geometry.clippedButtons, []);
      report.directViewports.push({locale, tab: index, ...geometry});
    }
  }
  await page.setViewportSize({width: 1440, height: 1000});
  for (const index of [1, 2, 3]) {
    await tab(index);
    const prefix = index === 1 ? '' : `pass_${index}_`;
    assert.equal(await page.locator(`#forge_nr_direct_${prefix}stage`).isVisible(), false);
    await page.locator(`#forge_nr_direct_pass_${index}_enabled input`).uncheck();
  }
  await run.click();
  await page.waitForFunction(() => /至少启用|Enable at least one/.test(document.querySelector('#forge_nr_direct_status textarea')?.value || ''));
  assert.equal(await page.locator('#forge_nr_direct_output img').count(), 0);
  await tab(1);
  await page.locator('#forge_nr_direct_pass_1_enabled input').check();
  await succeed();
  await page.getByText('CPU fixture', {exact: true}).click();
  const mode = async value => {
    await page.locator('#fixture_direct_mode').getByRole('radio', {name: value, exact: true}).check();
    await page.waitForFunction(expected => document.querySelector('#fixture_direct_mode_state textarea')?.value === expected, value);
  };
  await mode('wait');
  await run.click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_direct_cancel')?.disabled === false);
  assert.equal(await run.isDisabled(), true);
  assert.equal(await page.locator('#forge_nr_direct_output img').count(), 0);
  await cancel.click();
  await page.waitForFunction(() => /已取消|Canceled/.test(document.querySelector('#forge_nr_direct_status textarea')?.value || ''));
  assert.equal(await run.isDisabled(), false);
  assert.equal(await cancel.isDisabled(), true);
  assert.equal(await page.locator('#forge_nr_direct_output img').count(), 0);
  await mode('normal');
  await succeed();
  await mode('failure');
  await run.click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_direct_status textarea')?.value.includes('Synthetic NR failure'));
  assert.equal(await page.locator('#forge_nr_direct_output img').count(), 0);
  assert.equal(await run.isDisabled(), false);
  assert.equal(await page.locator('#forge_nr_direct_original img').getAttribute('src'), sourceUrl);
  await mode('normal');
  await succeed();
  await page.locator('#forge_nr_direct_input').getByRole('button', {name: 'Clear', exact: true}).click();
  await page.waitForFunction(() => {
    const source = document.querySelector('#forge_nr_direct_input');
    const preview = document.querySelector('#forge_nr_direct_original');
    return source?.getBoundingClientRect().height >= 359 && (!preview || !preview.getClientRects().length);
  });
  assert.equal(await page.locator('#forge_nr_direct_output img').count(), 0);
  assert.equal(await status.inputValue(), '');
  assert.ok(Math.abs((await sourceArea.boundingBox()).height - emptySource.height) <= 1);
  await page.getByRole('tab', {name: 'txt2img', exact: true}).click();
  assert.deepEqual(await capture(false), txtArgs);
  await page.getByRole('tab', {name: 'img2img', exact: true}).click();
  assert.deepEqual(await capture(false, true), imgArgs);
  report.direct.push({locale, upload: true, originalUnchanged: true, pngDownloadReceipt: true, rgbaPixels: true,
    repeatedRequestIdentities: true, threePasses: true, noPassRejected: true, cancel: true, failureClearsOutput: true, independentTabs: true,
    singleSourceArea: true, stableSourceHeight: true, clearRestoresUpload: true, compactPresetCrud: true});
}

try {
  if (process.argv.includes('--multipass-only')) {
    for (const locale of ['en', 'zh']) await multipass(locale);
  } else if (process.argv.includes('--presets-only')) {
    for (const locale of ['en', 'zh']) {
      await page.goto(`${origin}/direct-${locale}/`, {waitUntil: 'load'});
      await page.waitForFunction(() => document.querySelector('#forge_nr_direct_runtime textarea')?.value.includes('runtime_id'));
      await page.getByRole('tab', {name: 'DLSS5 NR', exact: true}).click();
      await presetToolbar('forge_nr_direct', locale);
      await presetCrud(locale);
      report.languages.push(locale);
    }
    await page.goto(`${origin}/direct-en/?without-presets-js`, {waitUntil: 'load'});
    await page.getByRole('tab', {name: 'DLSS5 NR', exact: true}).click();
    await page.waitForFunction(() => [...document.querySelectorAll('#forge_nr_direct_preset_toolbar button img')]
      .filter(image => image.complete && image.naturalWidth > 0 && image.currentSrc.startsWith('data:image/svg+xml;base64,')).length === 2);
    assert.equal(await page.evaluate(() => typeof window.forgeNRPresets), 'undefined');
    await page.locator('#forge_nr_direct_preset_toolbar').screenshot({path: path.join(folder, 'icons-without-script.png')});
    report.nativeIconsWithoutScript = true;
  } else {
  if (!process.argv.includes('--direct-only')) {
  await page.goto(`${origin}/new/`, {waitUntil: 'load'});
  await page.waitForFunction(() => document.querySelector('#forge_nr_setup_message')?.innerText.includes('Runtime download did not finish'));
  await page.locator('#forge_nr > .label-wrap').click();
  assert.doesNotMatch(await page.locator('#forge_nr_setup_message').innerText(), /Synthetic|pip/);
  assert.equal(await page.locator('#forge_nr_setup_details').isVisible(), false);
  await page.locator('#forge_nr').screenshot({path: path.join(folder, 'download-failed.png'), animations: 'disabled'});
  assert.equal(await page.locator('#forge_nr_permission').count(), 0);
  assert.equal(await page.locator('#forge_nr_runtime_file input[type="file"]').count(), 1);
  await page.locator('#forge_nr_prepare_runtime').click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_environment')?.innerText.includes('GPU detection finished'));
  assert.equal(await page.locator('#forge_nr_device input').inputValue(), '');
  await page.locator('#forge_nr_device input').click();
  await page.getByRole('option', {name: /Synthetic GPU/}).click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_environment')?.innerText.includes('Ready to generate'));
  assert.match(await page.locator('#forge_nr_setup_message').innerText(), /Ready to generate/);
  await page.locator('#forge_nr-visible-checkbox').check();
  assert.equal((await capture(true)).length, 35);
  assert.deepEqual(JSON.parse(await page.locator('#fixture_request textarea').inputValue()).preparation_calls, [false, false]);
  report.preparationRetry = true;

  await page.goto(`${origin}/repair/`, {waitUntil: 'load'});
  await page.waitForFunction(() => document.querySelector('#forge_nr_setup_message')?.innerText.includes('缺少依赖'));
  await page.locator('#forge_nr > .label-wrap').click();
  await page.locator('#forge_nr-visible-checkbox').uncheck();
  await page.waitForFunction(() => document.querySelector('#forge_nr-checkbox input')?.checked === false);
  await page.locator('#forge_nr_prepare_runtime').click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_setup_details textarea')?.value.includes('(attempt 2)'));
  await capture(false);
  assert.deepEqual(JSON.parse(await page.locator('#fixture_request textarea').inputValue()).preparation_calls, [false, false]);
  assert.match(await page.locator('#forge_nr_setup_message').innerText(), /修复依赖并继续/);
  await page.getByText('高级：诊断与修复', {exact: true}).click();
  await page.locator('#forge_nr_repair_dependencies').click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_setup_message')?.innerText.includes('依赖修复未完成'));
  assert.doesNotMatch(await page.locator('#forge_nr_setup_message').innerText(), /pip|CalledProcessError|CPU-fixture/);
  assert.match(await page.locator('#forge_nr_setup_details textarea').inputValue(), /CalledProcessError/);
  await page.locator('#forge_nr').screenshot({path: path.join(folder, 'repair-failed.png'), animations: 'disabled'});
  await page.locator('#forge_nr_repair_dependencies').click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_runtime textarea')?.value.includes('runtime_id')
    && document.querySelector('#forge_nr_setup_message')?.innerText.includes('列出显卡'));
  assert.equal(await page.locator('#forge_nr_setup_details textarea').inputValue(), '');
  await page.locator('#forge_nr_list_devices').click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_environment')?.innerText.includes('显卡检测完成'));
  await page.locator('#forge_nr_device input').click();
  await page.getByRole('option', {name: /Synthetic GPU/}).click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_setup_message')?.innerText.includes('已准备好'));
  await page.locator('#forge_nr-visible-checkbox').check();
  await capture(true);
  assert.deepEqual(JSON.parse(await page.locator('#fixture_request textarea').inputValue()).preparation_calls, [false, false, true, true]);
  report.explicitDependencyRepair = true;
  report.commandsOnlyInDiagnostics = true;
  let baseline;
  for (const [locale, label] of [['en', 'Insertion point'], ['zh', '插入时机']]) {
    await page.setViewportSize({width: 1440, height: 1000});
    await page.goto(`${origin}/${locale}/`, {waitUntil: 'load'});
    const header = page.locator('#forge_nr > .label-wrap');
    const toggle = page.locator('#forge_nr-visible-checkbox');
    await toggle.waitFor({state: 'visible'});
    await page.waitForFunction(() => document.querySelector('#forge_nr_runtime textarea')?.value.includes('runtime_id'));
    assert.equal(await page.locator('#forge_nr_language').count(), 0);
    await toggle.uncheck();
    await page.waitForFunction(() => document.querySelector('#forge_nr-checkbox input')?.checked === false);
    const args = await capture(false);
    baseline ??= args;
    assert.deepEqual(args, baseline);
    await header.click();
    await page.locator('#forge_nr_stage').waitFor({state: 'visible'});
    assert.ok((await page.locator('#forge_nr_stage').innerText()).includes(label));
    assert.match(await header.innerText(), /DLSS5 NR/);
    assert.doesNotMatch(await page.locator('body').innerText(), /tagsystem|DLSS NR/i);
    await page.locator('#forge_nr').screenshot({path: path.join(folder, `ui-${locale}.png`), animations: 'disabled'});
    await header.click();
    for (const [width, height] of [[1440,1000], [768,900], [390,844], [320,720]]) {
      await page.setViewportSize({width, height});
      await toggle.scrollIntoViewIfNeeded();
      assert.equal(await header.evaluate(element => element.classList.contains('open')), false);
      await toggle.check();
      await page.waitForFunction(() => document.querySelector('#forge_nr-checkbox input')?.checked === true);
      assert.equal(await header.evaluate(element => element.classList.contains('open')), false);
      assert.deepEqual((await capture(true)).slice(1), baseline.slice(1));
      await toggle.focus();
      await page.keyboard.press('Space');
      await page.waitForFunction(() => document.querySelector('#forge_nr-checkbox input')?.checked === false);
      assert.deepEqual(await capture(false), baseline);
      await header.click();
      await page.locator('#forge_nr_stage').waitFor({state: 'visible'});
      await page.locator('#forge_nr').screenshot({path: path.join(folder, `${locale}-${width}.png`), animations: 'disabled'});
      const geometry = await page.evaluate(() => ({width: innerWidth, height: innerHeight, documentWidth: document.documentElement.scrollWidth}));
      assert.ok(geometry.documentWidth <= geometry.width, 'Horizontal overflow');
      for (const selector of ['#forge_nr_setup_message', '#forge_nr_environment']) {
        const text = await page.locator(selector).evaluate(element => ({scroll: element.scrollHeight, visible: element.clientHeight}));
        assert.ok(text.scroll <= text.visible + 1, 'Readiness message must not be clipped');
      }
      report.viewports.push({locale, ...geometry});
      await header.click();
    }
    report.languages.push(locale);
  }
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto(`${origin}/zh/`, {waitUntil: 'load'});
  await page.locator('#forge_nr-visible-checkbox').waitFor({state: 'visible'});
  await page.waitForFunction(() => document.querySelector('#forge_nr_runtime textarea')?.value.includes('runtime_id'));
  await page.locator('#forge_nr-visible-checkbox').uncheck();
  await page.waitForFunction(() => document.querySelector('#forge_nr-checkbox input')?.checked === false);
  await capture(false);
  const automaticBefore = JSON.parse(await page.locator('#fixture_request textarea').inputValue()).automatic_preparations;
  await page.locator('#forge_nr > .label-wrap').click();
  await page.locator('#forge_nr-visible-checkbox').uncheck();
  await page.waitForFunction(() => document.querySelector('#forge_nr-checkbox input')?.checked === false);
  await page.locator('#forge_nr_runtime_mode').getByRole('radio', {name: '手动自定义／社区版', exact: true}).check();
  await page.waitForFunction(() => document.querySelector('#forge_nr_setup_message')?.innerText.includes('手动模式缺少'));
  assert.match(await page.locator('#forge_nr_runtime_note').innerText(), /不会下载或覆盖/);
  await capture(false);
  assert.equal(JSON.parse(await page.locator('#fixture_request textarea').inputValue()).automatic_preparations, automaticBefore);
  await page.getByText('高级：诊断与修复', {exact: true}).click();
  const manualFile = Buffer.alloc(2048);
  manualFile.write('MZ', 0, 'ascii');
  manualFile.writeUInt32LE(128, 60);
  manualFile.write('PE\0\0', 128, 'ascii');
  manualFile.writeUInt16LE(0x8664, 132);
  manualFile.writeUInt16LE(0x2000, 150);
  manualFile.writeUInt16LE(0x20b, 152);
  manualFile.write('CPU_FIXTURE_ONLY', 512, 'ascii');
  const uploaded = page.waitForResponse(response => response.url().includes('/upload') && response.request().method() === 'POST');
  await page.locator('#forge_nr_runtime_file input[type="file"]').setInputFiles({name: 'nvngx_dlssnr.dll', mimeType: 'application/octet-stream', buffer: manualFile});
  const uploadResponse = await uploaded;
  assert.equal(uploadResponse.ok(), true);
  await page.locator('#forge_nr_runtime_file [aria-label="nvngx_dlssnr.dll"]').waitFor({state: 'visible'});
  await page.locator('#forge_nr_import_runtime').click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_environment')?.innerText.includes('手动文件格式已检查'));
  const manualArgs = await capture(false);
  const imported = JSON.parse(await page.locator('#fixture_request textarea').inputValue());
  assert.equal(imported.manual_imported, true);
  assert.equal(imported.automatic_preparations, automaticBefore);
  assert.match(JSON.parse(manualArgs[11]).runtime_dir, /[/\\]manual$/);
  await page.locator('#forge_nr').screenshot({path: path.join(folder, 'manual-imported.png'), animations: 'disabled'});
  await page.reload({waitUntil: 'load'});
  await page.waitForFunction(() => document.querySelector('#forge_nr_runtime_note')?.innerText.includes('手动模式不会下载'));
  await page.waitForFunction(() => document.querySelector('#forge_nr_environment')?.innerText.includes('手动文件格式已检查'));
  await page.locator('#forge_nr > .label-wrap').click();
  await page.locator('#forge_nr-visible-checkbox').uncheck();
  await page.waitForFunction(() => document.querySelector('#forge_nr-checkbox input')?.checked === false);
  assert.equal(await page.locator('#forge_nr_runtime_mode').getByRole('radio', {name: '手动自定义／社区版', exact: true}).isChecked(), true);
  assert.deepEqual(await capture(false), manualArgs);
  assert.equal(JSON.parse(await page.locator('#fixture_request textarea').inputValue()).automatic_preparations, automaticBefore);
  await page.locator('#forge_nr_runtime_mode').getByRole('radio', {name: '自动下载原版', exact: true}).check();
  await page.waitForFunction(() => {
    try { return JSON.parse(document.querySelector('#forge_nr_runtime textarea').value).runtime_id === 'a'.repeat(64); }
    catch { return false; }
  });
  const automaticArgs = await capture(false);
  assert.deepEqual(automaticArgs.slice(0, 11), manualArgs.slice(0, 11));
  assert.doesNotMatch(JSON.parse(automaticArgs[11]).runtime_dir, /[/\\]manual$/);
  assert.equal(JSON.parse(await page.locator('#fixture_request textarea').inputValue()).manual_imported, true);
  report.manualRuntimeUpload = true;
  report.manualModeSurvivesRefresh = true;
  report.manualModeSkipsAutomaticPreparation = true;
  for (const locale of ['en', 'zh']) await multipass(locale);
  for (const locale of ['en', 'zh']) await img2img(locale);
  }
  for (const locale of ['en', 'zh']) await direct(locale);
  }
  assert.deepEqual(report.errors, []);
  report.passed = true;
} catch (error) {
  report.failure = String(error.stack || error);
  report.presetFailure = await page.evaluate(() => [...document.querySelectorAll('.forge-nr-preset-toolbar')].map(toolbar => {
    const prefix = toolbar.id.replace(/_preset_toolbar$/, '');
    return {prefix, name: toolbar.querySelector('input')?.value,
      selection: document.querySelector(`#${prefix}_preset_selection textarea`)?.value,
      message: document.querySelector(`#${prefix}_preset_message`)?.textContent};
  })).catch(() => []);
  await page.screenshot({path: path.join(folder, 'failure.png'), fullPage: true}).catch(() => {});
  process.exitCode = 1;
} finally {
  await context.close();
  await browser.close();
  await writeFile(path.join(folder, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}