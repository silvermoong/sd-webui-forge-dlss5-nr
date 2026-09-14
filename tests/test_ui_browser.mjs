import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const origin = 'http://127.0.0.1:7875';
const folder = path.join(root, 'work', `ui-browser-${Date.now()}`);
await mkdir(folder, {recursive: true});
const report = {folder, errors: [], viewports: [], languages: [], multipass: [], passViewports: []};
const browser = await chromium.launch({channel: 'msedge', headless: true,
  args: ['--disable-gpu', '--disable-background-networking', '--disable-component-update', '--disable-sync']});
const context = await browser.newContext({viewport: {width: 1440, height: 1000}, serviceWorkers: 'block'});
await context.route('**/*', route => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
const page = await context.newPage();
page.on('pageerror', error => report.errors.push(String(error)));

async function capture(enabled) {
  const previous = await page.locator('#fixture_request textarea').inputValue();
  const sequence = previous ? JSON.parse(previous).sequence : 0;
  await page.locator('#fixture_capture').click();
  await page.waitForFunction(expected => {
    try {
      const value = JSON.parse(document.querySelector('#fixture_request textarea').value);
      return value.sequence > expected.sequence && value.args[0] === expected.enabled;
    } catch { return false; }
  }, {sequence, enabled});
  const result = JSON.parse(await page.locator('#fixture_request textarea').inputValue());
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
  await page.getByText(locale === 'zh' ? '本地命名参数预设' : 'Named parameter presets', {exact: true}).click();
  await page.locator('#forge_nr_preset_name').getByRole('textbox').fill(`three-pages-${locale}`);
  await page.locator('#forge_nr_save_preset').click();
  await page.waitForFunction(() => /已保存整组三页|All three tabs saved/.test(
    document.querySelector('#forge_nr_preset_message textarea')?.value || ''));
  await tab(3);
  await number(3, 'mix', .1);
  await enabled(3).uncheck();
  await master.uncheck();
  await page.locator('#forge_nr_load_preset').click();
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
  await page.locator('#forge_nr_load_preset').click();
  await page.waitForFunction(() => [1, 2, 3].every(index =>
    document.querySelector(`#forge_nr_pass_${index}_enabled input`)?.checked === true));
  assert.deepEqual((await snapshot()).spec, saved.spec);
  for (const [width, height] of [[1440,1000], [1040,900], [768,900], [390,844], [320,720], [1440,540]]) {
    await page.setViewportSize({width, height});
    for (const index of [1, 2, 3]) {
      await tab(index);
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
  await page.locator('#forge_nr_load_preset').click();
  await page.waitForFunction(() => document.querySelector('#forge_nr_pass_3_enabled input')?.checked === false);
  record = await snapshot();
  assert.equal(record.spec.params.mix, .25);
  assert.equal(record.spec.stage, 'before_hr');
  assert.equal(record.execution_order.length, 1);
  assert.equal(record.args[11], initial.args[11]);
  report.multipass.push({locale, independentParams: true, disabledSkip: true, allDisabled: true,
    stageOrder: [2, 1, 3], copyKeepsDisabled: true, presets: true, legacyMigration: true});
}

try {
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
  assert.deepEqual(report.errors, []);
  report.passed = true;
} catch (error) {
  report.failure = String(error.stack || error);
  await page.screenshot({path: path.join(folder, 'failure.png'), fullPage: true}).catch(() => {});
  process.exitCode = 1;
} finally {
  await context.close();
  await browser.close();
  await writeFile(path.join(folder, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}